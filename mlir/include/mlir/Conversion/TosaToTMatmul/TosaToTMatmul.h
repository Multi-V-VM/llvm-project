//===- TosaToTMatmul.h - TOSA to TMatmul conversion ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOTMATMUL_TOSATOTMATMUL_H
#define MLIR_CONVERSION_TOSATOTMATMUL_TOSATOTMATMUL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;

namespace tosa {
/// Populates conversion patterns for lowering TOSA ops to TMatmul ops.
void populateTosaToTMatmulConversionPatterns(RewritePatternSet &patterns);

/// Creates a pass to convert TOSA operations to TMatmul dialect.
std::unique_ptr<Pass> createTosaToTMatmulPass();

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOTMATMUL_TOSATOTMATMUL_H
