//===- TMatmulOps.td - TMatmul dialect operations ----------*- tablegen -*-===//
//
// TMatmul Operations - Custom ISA operations
//
//===----------------------------------------------------------------------===//

#ifndef TMATMUL_OPS
#define TMATMUL_OPS

include "mlir/Dialect/TMatmul/IR/TMatmulDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def TMatmul_LoadVectorOp : TMatmul_Op<"ldv", [Pure]> {
  let summary = "Load vector from DDR memory";
  let description = [{
    Loads a vector from DDR memory location into a vector register.
    Syntax: ldv vX, MEM_LOCATION

    Example:
    ```mlir
    %v0 = tmatmul.ldv %X : tensor<512xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$source attr-dict `:` type($source)";
}

def TMatmul_StoreVectorOp : TMatmul_Op<"sv"> {
  let summary = "Store vector to DDR memory";
  let description = [{
    Stores a vector register to DDR memory location.
    Syntax: sv vX, MEM_LOCATION

    Example:
    ```mlir
    tmatmul.sv %v0, %oH : tensor<512xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$source, AnyTensor:$dest);

  let assemblyFormat = "$source `,` $dest attr-dict `:` type($source) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def TMatmul_AddOp : TMatmul_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Vector addition";
  let description = [{
    Element-wise vector addition: result = lhs + rhs
    Syntax: add vDst, vSrc1, vSrc2
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TMatmul_SubOp : TMatmul_Op<"sub", [Pure, SameOperandsAndResultType]> {
  let summary = "Vector subtraction";
  let description = [{
    Element-wise vector subtraction: result = lhs - rhs
    Syntax: sub vDst, vSrc1, vSrc2
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TMatmul_MulOp : TMatmul_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "Vector multiplication";
  let description = [{
    Element-wise vector multiplication: result = lhs * rhs
    Syntax: mul vDst, vSrc1, vSrc2
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TMatmul_DivOp : TMatmul_Op<"div", [Pure, SameOperandsAndResultType]> {
  let summary = "Vector division";
  let description = [{
    Element-wise vector division: result = lhs / rhs
    Syntax: div vDst, vSrc1, vSrc2
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Activation Functions
//===----------------------------------------------------------------------===//

def TMatmul_SigmoidOp : TMatmul_Op<"sig", [Pure, SameOperandsAndResultType]> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid function: result = 1 / (1 + exp(-input))
    Syntax: sig vDst, vSrc
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def TMatmul_ComplementSigmoidOp : TMatmul_Op<"csig", [Pure, SameOperandsAndResultType]> {
  let summary = "Complement sigmoid activation";
  let description = [{
    Applies complement sigmoid: result = 1 - sigmoid(input)
    Syntax: csig vDst, vSrc
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def TMatmul_SiLUOp : TMatmul_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "SiLU (Swish) activation";
  let description = [{
    Applies SiLU activation: result = x * sigmoid(x)
    Syntax: silu vDst, vSrc
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Special Operations
//===----------------------------------------------------------------------===//

def TMatmul_NormOp : TMatmul_Op<"norm", [Pure, SameOperandsAndResultType]> {
  let summary = "Layer normalization";
  let description = [{
    Applies layer normalization to the input vector.
    Syntax: norm vDst, vSrc
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def TMatmul_ImportOp : TMatmul_Op<"tmatmul_import"> {
  let summary = "Import vector into tmatmul unit";
  let description = [{
    Loads a vector into the tmatmul accelerator's internal register.
    Syntax: tmatmul_import vSrc
  }];

  let arguments = (ins AnyTensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def TMatmul_GoOp : TMatmul_Op<"tmatmul_go"> {
  let summary = "Execute matrix multiplication";
  let description = [{
    Initiates matrix multiplication with the specified weight matrix.
    The input is the previously imported vector.
    Syntax: tmatmul_go WEIGHT_MATRIX

    Example:
    ```mlir
    tmatmul.tmatmul_go %WF : tensor<512x512xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$weights);

  let assemblyFormat = "$weights attr-dict `:` type($weights)";
}

def TMatmul_ExportOp : TMatmul_Op<"tmatmul_export", [Pure]> {
  let summary = "Export result from tmatmul unit";
  let description = [{
    Exports the result of the matrix multiplication to a vector register.
    Syntax: tmatmul_export vDst
  }];

  let results = (outs AnyTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Combined TMatmul Operation (for optimization)
//===----------------------------------------------------------------------===//

def TMatmul_MatMulOp : TMatmul_Op<"matmul", [Pure]> {
  let summary = "Complete matrix multiplication sequence";
  let description = [{
    Combined operation representing import->go->export sequence.
    This can be lowered to the three separate operations or handled
    as a single fused operation for optimization.

    Example:
    ```mlir
    %result = tmatmul.matmul %input, %weights : tensor<512xf32>, tensor<512x512xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$weights);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights attr-dict `:` type($input) `,` type($weights) `->` type($result)";
}

#endif // TMATMUL_OPS
