// ============================================================================
// File: shakti_ai_inference.c
// Target: SHAKTI RISC-V Core (RV32IMAC / RV64GC)
// Compiler: riscv64-unknown-elf-gcc -march=rv32imac -mabi=ilp32 -O2
// ============================================================================

#include <stdint.h>
#include <stdio.h>

#define FPGA_ACCEL_BASE   0x40020000UL

// Memory-Mapped Volatile Register Access
#define REG_CTRL          (*(volatile uint32_t *)(FPGA_ACCEL_BASE + 0x00))
#define REG_STATUS        (*(volatile uint32_t *)(FPGA_ACCEL_BASE + 0x04))
#define REG_MAT_A_ROW(r)  (*(volatile uint32_t *)(FPGA_ACCEL_BASE + 0x10 + ((r) * 4)))
#define REG_MAT_B_ROW(r)  (*(volatile uint32_t *)(FPGA_ACCEL_BASE + 0x20 + ((r) * 4)))
#define REG_MAT_C_VAL(i)  (*(volatile int32_t  *)(FPGA_ACCEL_BASE + 0x30 + ((i) * 4)))

// Macro: Pack 4 signed INT8 values into a single 32-bit AXI word
#define PACK_INT8x4(b0, b1, b2, b3) \
    ((((uint32_t)(uint8_t)(b0))      ) | \
     (((uint32_t)(uint8_t)(b1)) <<  8) | \
     (((uint32_t)(uint8_t)(b2)) << 16) | \
     (((uint32_t)(uint8_t)(b3)) << 24))

// Read RISC-V Hardware Cycle Counter CSR
static inline uint32_t read_rdcycle(void) {
    uint32_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// Hardware Offload Function
void gemm_fpga_offload(const int8_t A[4][4], const int8_t B[4][4], int32_t C[4][4]) {
    // 1. Pack and write Matrix A & B over AXI4-Lite
    for (int r = 0; r < 4; r++) {
        REG_MAT_A_ROW(r) = PACK_INT8x4(A[r][0], A[r][1], A[r][2], A[r][3]);
        REG_MAT_B_ROW(r) = PACK_INT8x4(B[r][0], B[r][1], B[r][2], B[r][3]);
    }

    // 2. Trigger FPGA
    REG_CTRL = 0x01;

    // 3. Poll until DONE bit (Bit 1) is high
    while ((REG_STATUS & 0x02) == 0) {
        // Wait for 4-cycle systolic computation
    }

    // 4. Read back 16 accumulated INT32 outputs
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[i][j] = REG_MAT_C_VAL(i * 4 + j);
        }
    }
}

// Software Control: Bias Addition, ReLU Activation, Argmax Classification
void ai_post_processing_shakti(int32_t C[4][4], const int32_t bias[4], int *predicted_class) {
    int32_t class_scores[4] = {0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int32_t val = C[i][j] + bias[j];
            C[i][j] = (val > 0) ? val : 0; // ReLU non-linearity
            class_scores[j] += C[i][j];    // Pooling
        }
    }

    int best_idx = 0;
    for (int j = 1; j < 4; j++) {
        if (class_scores[j] > class_scores[best_idx]) {
            best_idx = j;
        }
    }
    *predicted_class = best_idx;
}

int main(void) {
    const int8_t A[4][4] = {
        {2, 1, 0, 3}, {1, 4, 2, 1}, {3, 0, 1, 2}, {2, 2, 3, 1}
    };
    const int8_t B[4][4] = {
        { 1, -1,  2,  0}, { 2,  3, -1,  1}, { 0,  2,  1, -2}, { 1,  0,  3,  2}
    };
    const int32_t bias[4] = {1, 0, -2, 2};
    int32_t output_hw[4][4];
    int predicted_class = -1;

    uint32_t t_start = read_rdcycle();
    gemm_fpga_offload(A, B, output_hw);
    ai_post_processing_shakti(output_hw, bias, &predicted_class);
    uint32_t t_end = read_rdcycle();

    printf("AI Inference Complete! Predicted Class = %d (Total Cycles: %lu)\n",
           predicted_class, (unsigned long)(t_end - t_start));
    return 0;
}
