//===- TMatmulDialect.h - TMatmul dialect -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TMATMUL_IR_TMATMULDIALECT_H
#define MLIR_DIALECT_TMATMUL_IR_TMATMULDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/TMatmul/IR/TMatmulOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/TMatmul/IR/TMatmulOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/TMatmul/IR/TMatmulOps.h.inc"

#endif // MLIR_DIALECT_TMATMUL_IR_TMATMULDIALECT_H
