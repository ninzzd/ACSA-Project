// tb_ingress -- exhaustive check of ipu_ingress over all 65,536 fp16 inputs.
`timescale 1ns/1ps
`ifndef VEC_DIR
`define VEC_DIR "../vectors/"
`endif
module tb_ingress;
  localparam int P = 16;
  logic [17:0]     exp_mem [0:65535];
  logic [16*P-1:0] fp16;
  logic [17*P-1:0] delta;
  logic [P-1:0]    bad;
  logic [16*P-1:0] tmp;
  int errors = 0;

  ipu_ingress #(.P(P)) dut (.fp16(fp16), .delta(delta), .bad(bad));

  initial begin
    $readmemh({`VEC_DIR, "unit/ingress_exp.hex"}, exp_mem);
    if ($isunknown(exp_mem[65535])) $fatal(1, "vector file missing");
    for (int base = 0; base < 65536; base += P) begin
      for (int i = 0; i < P; i++) tmp[16*i +: 16] = 16'(base + i);
      fp16 = tmp;
      #1;
      for (int i = 0; i < P; i++) begin
        if ({bad[i], delta[17*i +: 17]} !== exp_mem[base + i]) begin
          if (errors < 10)
            $display("MISMATCH fp16=%04h got bad=%0d q=%05h exp %05h", base + i, bad[i],
                     delta[17*i +: 17], exp_mem[base + i]);
          errors++;
        end
      end
    end
    if (errors == 0) $display("PASS tb_ingress: 65536 inputs");
    else             $display("FAIL tb_ingress: %0d mismatches", errors);
    $finish;
  end
endmodule
