# SHAKTI-FPGA-AI-Inference-Hardware-Software-Co-Design-Project
# Hardware/Software Co-Design for AI Inference Using SHAKTI RISC-V + FPGA

An open-source Hardware/Software Co-Design system that offloads General Matrix Multiplication (GEMM) from the **SHAKTI RISC-V Processor** to a custom **4x4 Parallel Systolic Array** on a **Xilinx Artix-7 FPGA** via an **AMBA AXI4-Lite** bus (`0x40020000`).

## Performance Results

| Metric | SHAKTI SW-Only | SHAKTI + FPGA Co-Design | Speedup |
| :--- | :--- | :--- | :--- |
| **Matrix Multiply (64 MACs)** | 576 cycles | 4 cycles | **144.0x** |
| **AXI4-Lite I/O Overhead** | 0 cycles | 40 cycles | N/A |
| **Software ReLU + Argmax** | 36 cycles | 24 cycles | 1.5x |
| **Total 4x4 Layer Latency** | 612 cycles | 68 cycles | **9.0x** |

## How to Run RTL Simulation
```bash
cd rtl
iverilog -o sim_out gemm_axi_accelerator.v tb_gemm_axi.v
vvp sim_out
