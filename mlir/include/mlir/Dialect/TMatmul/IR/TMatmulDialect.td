//===- TMatmulDialect.td - TMatmul dialect definition ------*- tablegen -*-===//
//
// TMatmul Dialect - Custom ISA for matrix multiply accelerator
//
//===----------------------------------------------------------------------===//

#ifndef TMATMUL_DIALECT
#define TMATMUL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// TMatmul Dialect Definition
//===----------------------------------------------------------------------===//

def TMatmul_Dialect : Dialect {
  let name = "tmatmul";
  let summary = "TMatmul accelerator dialect";
  let description = [{
    This dialect represents operations for a custom matrix multiplication
    accelerator with 8 vector registers (v0-v7) and specialized operations
    for tensor matmul, normalization, and activation functions.

    The ISA includes:
    - Memory operations: ldv (load vector), sv (save vector)
    - Arithmetic: add, sub, mul, div
    - Activations: sig (sigmoid), csig (complement sigmoid), silu
    - Special: norm, tmatmul_import, tmatmul_go, tmatmul_export
  }];

  let cppNamespace = "::mlir::tmatmul";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// TMatmul Type Definitions
//===----------------------------------------------------------------------===//

def TMatmul_VectorType : TypeDef<TMatmul_Dialect, "Vector"> {
  let mnemonic = "vector";
  let summary = "TMatmul vector type";
  let description = [{
    Represents a vector stored in DDR memory or vector registers.
    Vectors have dynamic size determined at runtime.
  }];

  let parameters = (ins "Type":$elementType, "int64_t":$size);
  let assemblyFormat = "`<` $size `x` $elementType `>`";
}

def TMatmul_MemRefType : AnyTypeOf<[AnyMemRef]>;

//===----------------------------------------------------------------------===//
// TMatmul Base Operation
//===----------------------------------------------------------------------===//

class TMatmul_Op<string mnemonic, list<Trait> traits = []> :
    Op<TMatmul_Dialect, mnemonic, traits>;

#endif // TMATMUL_DIALECT
