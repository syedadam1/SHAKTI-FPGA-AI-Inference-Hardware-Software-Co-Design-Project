// ============================================================================
// Module: gemm_axi_accelerator.v
// Project: Hardware/Software Co-Design for AI Inference (SHAKTI + FPGA)
// Target: Xilinx Artix-7 FPGA / SHAKTI RISC-V SoC Interconnect
// Base Address: 0x4002_0000
// ============================================================================
`timescale 1ns / 1ps

module gemm_axi_accelerator (
    input  wire        S_AXI_ACLK,
    input  wire        S_AXI_ARESETN,

    // AXI4-Lite Write Address Channel
    input  wire [7:0]  S_AXI_AWADDR,
    input  wire        S_AXI_AWVALID,
    output reg         S_AXI_AWREADY,

    // AXI4-Lite Write Data Channel
    input  wire [31:0] S_AXI_WDATA,
    input  wire [3:0]  S_AXI_WSTRB,
    input  wire        S_AXI_WVALID,
    output reg         S_AXI_WREADY,

    // AXI4-Lite Write Response Channel
    output reg  [1:0]  S_AXI_BRESP,
    output reg         S_AXI_BVALID,
    input  wire        S_AXI_BREADY,

    // AXI4-Lite Read Address Channel
    input  wire [7:0]  S_AXI_ARADDR,
    input  wire        S_AXI_ARVALID,
    output reg         S_AXI_ARREADY,

    // AXI4-Lite Read Data Channel
    output reg  [31:0] S_AXI_RDATA,
    output reg  [1:0]  S_AXI_RRESP,
    output reg         S_AXI_RVALID,
    input  wire        S_AXI_RREADY
);

    // Register Storage (Offsets: 0x00=CTRL, 0x04=STATUS, 0x10..1C=MAT_A, 0x20..2C=MAT_B)
    reg [31:0] ctrl_reg;
    reg [31:0] mat_a_packed [0:3];
    reg [31:0] mat_b_packed [0:3];
    reg signed [31:0] mat_c [0:15];

    reg        busy;
    reg        done;
    reg [2:0]  k_step;
    reg [7:0]  cycle_counter;

    // Unpack 32-bit registers into 4x4 signed INT8 wires
    wire signed [7:0] A [0:3][0:3];
    wire signed [7:0] B [0:3][0:3];

    genvar r;
    generate
        for (r = 0; r < 4; r = r + 1) begin : UNPACK_BLOCK
            assign A[r][0] = mat_a_packed[r][7:0];
            assign A[r][1] = mat_a_packed[r][15:8];
            assign A[r][2] = mat_a_packed[r][23:16];
            assign A[r][3] = mat_a_packed[r][31:24];

            assign B[r][0] = mat_b_packed[r][7:0];
            assign B[r][1] = mat_b_packed[r][15:8];
            assign B[r][2] = mat_b_packed[r][23:16];
            assign B[r][3] = mat_b_packed[r][31:24];
        end
    endgenerate

    integer i, j;

    // AXI4-Lite Write Handshake
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_AWREADY <= 1'b0;
            S_AXI_WREADY  <= 1'b0;
            S_AXI_BVALID  <= 1'b0;
            S_AXI_BRESP   <= 2'b00;
            ctrl_reg      <= 32'd0;
            for (i = 0; i < 4; i = i + 1) begin
                mat_a_packed[i] <= 32'd0;
                mat_b_packed[i] <= 32'd0;
            end
        end else begin
            if (ctrl_reg[0]) ctrl_reg[0] <= 1'b0; // Auto-clear START pulse

            if (!S_AXI_AWREADY && S_AXI_AWVALID && S_AXI_WVALID) begin
                S_AXI_AWREADY <= 1'b1;
                S_AXI_WREADY  <= 1'b1;
                S_AXI_BVALID  <= 1'b1;
                S_AXI_BRESP   <= 2'b00; // OKAY

                case (S_AXI_AWADDR)
                    8'h00: ctrl_reg        <= S_AXI_WDATA;
                    8'h10: mat_a_packed[0] <= S_AXI_WDATA;
                    8'h14: mat_a_packed[1] <= S_AXI_WDATA;
                    8'h18: mat_a_packed[2] <= S_AXI_WDATA;
                    8'h1C: mat_a_packed[3] <= S_AXI_WDATA;
                    8'h20: mat_b_packed[0] <= S_AXI_WDATA;
                    8'h24: mat_b_packed[1] <= S_AXI_WDATA;
                    8'h28: mat_b_packed[2] <= S_AXI_WDATA;
                    8'h2C: mat_b_packed[3] <= S_AXI_WDATA;
                    default: ;
                endcase
            end else begin
                S_AXI_AWREADY <= 1'b0;
                S_AXI_WREADY  <= 1'b0;
                if (S_AXI_BVALID && S_AXI_BREADY) begin
                    S_AXI_BVALID <= 1'b0;
                end
            end
        end
    end

    // Parallel 4x4 Systolic MAC Engine: 16 Multipliers Compute in 4 Cycles
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN || ctrl_reg[1]) begin
            busy          <= 1'b0;
            done          <= 1'b0;
            k_step        <= 3'd0;
            cycle_counter <= 8'd0;
            for (i = 0; i < 16; i = i + 1) begin
                mat_c[i] <= 32'sd0;
            end
        end else begin
            if (ctrl_reg[0] && !busy) begin
                busy          <= 1'b1;
                done          <= 1'b0;
                k_step        <= 3'd0;
                cycle_counter <= 8'd0;
                for (i = 0; i < 16; i = i + 1) begin
                    mat_c[i] <= 32'sd0;
                end
            end else if (busy) begin
                cycle_counter <= cycle_counter + 8'd1;
                if (k_step < 3'd4) begin
                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin
                            mat_c[i*4 + j] <= mat_c[i*4 + j] + (A[i][k_step[1:0]] * B[k_step[1:0]][j]);
                        end
                    end
                    k_step <= k_step + 3'd1;
                end else begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    // AXI4-Lite Read Handshake
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_ARREADY <= 1'b0;
            S_AXI_RVALID  <= 1'b0;
            S_AXI_RRESP   <= 2'b00;
            S_AXI_RDATA   <= 32'd0;
        end else begin
            if (!S_AXI_ARREADY && S_AXI_ARVALID) begin
                S_AXI_ARREADY <= 1'b1;
                S_AXI_RVALID  <= 1'b1;
                S_AXI_RRESP   <= 2'b00;
                case (S_AXI_ARADDR)
                    8'h00: S_AXI_RDATA <= ctrl_reg;
                    8'h04: S_AXI_RDATA <= {16'd0, cycle_counter, 6'd0, done, busy};
                    default: begin
                        if (S_AXI_ARADDR >= 8'h30 && S_AXI_ARADDR <= 8'h6C) begin
                            S_AXI_RDATA <= mat_c[(S_AXI_ARADDR - 8'h30) >> 2];
                        end else begin
                            S_AXI_RDATA <= 32'd0;
                        end
                    end
                endcase
            end else begin
                S_AXI_ARREADY <= 1'b0;
                if (S_AXI_RVALID && S_AXI_RREADY) begin
                    S_AXI_RVALID <= 1'b0;
                end
            end
        end
    end

endmodule
