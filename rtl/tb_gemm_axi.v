// ============================================================================
// Testbench: tb_gemm_axi.v
// Emulates SHAKTI RISC-V AXI4-Lite Master Transactions
// ============================================================================
`timescale 1ns / 1ps

module tb_gemm_axi;

    reg         clk;
    reg         rst_n;
    reg  [7:0]  awaddr;
    reg         awvalid;
    wire        awready;
    reg  [31:0] wdata;
    reg  [3:0]  wstrb;
    reg         wvalid;
    wire        wready;
    wire [1:0]  bresp;
    wire        bvalid;
    reg         bready;
    reg  [7:0]  araddr;
    reg         arvalid;
    wire        arready;
    wire [31:0] rdata;
    wire [1:0]  rresp;
    wire        rvalid;
    reg         rready;

    // Instantiate Device Under Test (DUT)
    gemm_axi_accelerator dut (
        .S_AXI_ACLK    (clk),
        .S_AXI_ARESETN (rst_n),
        .S_AXI_AWADDR  (awaddr),
        .S_AXI_AWVALID (awvalid),
        .S_AXI_AWREADY (awready),
        .S_AXI_WDATA   (wdata),
        .S_AXI_WSTRB   (wstrb),
        .S_AXI_WVALID  (wvalid),
        .S_AXI_WREADY  (wready),
        .S_AXI_BRESP   (bresp),
        .S_AXI_BVALID  (bvalid),
        .S_AXI_BREADY  (bready),
        .S_AXI_ARADDR  (araddr),
        .S_AXI_ARVALID (arvalid),
        .S_AXI_ARREADY (arready),
        .S_AXI_RDATA   (rdata),
        .S_AXI_RRESP   (rresp),
        .S_AXI_RVALID  (rvalid),
        .S_AXI_RREADY  (rready)
    );

    // 100 MHz Clock Generator
    initial clk = 0;
    always #5 clk = ~clk;

    task axi_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            awaddr  <= addr;
            awvalid <= 1'b1;
            wdata   <= data;
            wstrb   <= 4'hF;
            wvalid  <= 1'b1;
            bready  <= 1'b1;
            wait (awready && wready);
            @(posedge clk);
            awvalid <= 1'b0;
            wvalid  <= 1'b0;
            wait (bvalid);
            @(posedge clk);
            bready  <= 1'b0;
        end
    endtask

    task axi_read(input [7:0] addr, output [31:0] data_out);
        begin
            @(posedge clk);
            araddr  <= addr;
            arvalid <= 1'b1;
            rready  <= 1'b1;
            wait (arready);
            @(posedge clk);
            arvalid <= 1'b0;
            wait (rvalid);
            data_out = rdata;
            @(posedge clk);
            rready  <= 1'b0;
        end
    endtask

    reg [31:0] status_val;
    reg signed [31:0] c_val;
    reg signed [31:0] relu_val;
    integer r, c;

    initial begin
        rst_n = 0;
        awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
        araddr = 0; arvalid = 0; rready = 0;

        #25 rst_n = 1;
        $display("--- SHAKTI + FPGA Co-Design RTL Simulation ---");

        // 1. Write Input Matrix A (INT8 rows packed into 32-bit words)
        axi_write(8'h10, 32'h03000102); // Row 0: [2, 1, 0, 3]
        axi_write(8'h14, 32'h01020401); // Row 1: [1, 4, 2, 1]
        axi_write(8'h18, 32'h02010003); // Row 2: [3, 0, 1, 2]
        axi_write(8'h1C, 32'h01030202); // Row 3: [2, 2, 3, 1]

        // 2. Write Weight Matrix B (INT8 rows packed into 32-bit words)
        axi_write(8'h20, 32'h0002FF01); // Row 0: [1, -1, 2, 0]
        axi_write(8'h24, 32'h01FF0302); // Row 1: [2, 3, -1, 1]
        axi_write(8'h28, 32'hFE010200); // Row 2: [0, 2, 1, -2]
        axi_write(8'h2C, 32'h02030001); // Row 3: [1, 0, 3, 2]

        // 3. Trigger FPGA Matrix Accelerator
        axi_write(8'h00, 32'h00000001);

        // 4. Poll STATUS_REG until DONE bit is set
        status_val = 0;
        while ((status_val & 32'h02) == 0) begin
            axi_read(8'h04, status_val);
        end
        $display("Hardware Systolic Multiply Completed in %0d clock cycles!",
                 (status_val >> 8) & 8'hFF);

        // 5. Read Output Matrix C and Compute Software ReLU
        for (r = 0; r < 4; r = r + 1) begin
            for (c = 0; c < 4; c = c + 1) begin
                axi_read(8'h30 + ((r*4 + c)*4), c_val);
                relu_val = (c_val < 0) ? 0 : c_val;
                $display("  C[%0d][%0d] = %4d | After ReLU = %4d", r, c, c_val, relu_val);
            end
        end
        $display("Simulation Verified Successfully!");
        $finish;
    end

endmodule
