//===- TosaToTMatmul.cpp - Lowering TOSA to TMatmul dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering patterns from TOSA ops to TMatmul ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToTMatmul/TosaToTMatmul.h"
#include "mlir/Dialect/TMatmul/IR/TMatmulDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

//===----------------------------------------------------------------------===//
// TOSA to TMatmul Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert tosa.add to tmatmul.add
struct TosaAddToTMatmulPattern : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tmatmul::AddOp>(op, op.getType(), op.getInput1(),
                                                 op.getInput2());
    return success();
  }
};

/// Convert tosa.sub to tmatmul.sub
struct TosaSubToTMatmulPattern : public OpRewritePattern<tosa::SubOp> {
  using OpRewritePattern<tosa::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SubOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tmatmul::SubOp>(op, op.getType(), op.getInput1(),
                                                 op.getInput2());
    return success();
  }
};

/// Convert tosa.mul to tmatmul.mul
struct TosaMulToTMatmulPattern : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tmatmul::MulOp>(op, op.getType(), op.getInput1(),
                                                 op.getInput2());
    return success();
  }
};

/// Convert tosa.sigmoid to tmatmul.sig
struct TosaSigmoidToTMatmulPattern : public OpRewritePattern<tosa::SigmoidOp> {
  using OpRewritePattern<tosa::SigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tmatmul::SigmoidOp>(op, op.getType(),
                                                     op.getInput1());
    return success();
  }
};

/// Convert tosa.fully_connected to tmatmul.matmul
/// This handles matrix multiplication which maps to the tmatmul accelerator
struct TosaFullyConnectedToTMatmulPattern
    : public OpRewritePattern<tosa::FullyConnectedOp> {
  using OpRewritePattern<tosa::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::FullyConnectedOp op,
                                PatternRewriter &rewriter) const override {
    // Get input and weight tensors
    Value input = op.getInput();
    Value weight = op.getWeight();

    // Create tmatmul.matmul operation
    rewriter.replaceOpWithNewOp<tmatmul::MatMulOp>(op, op.getType(), input,
                                                    weight);
    return success();
  }
};

/// Convert tosa.matmul to tmatmul.matmul
struct TosaMatMulToTMatmulPattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tmatmul::MatMulOp>(op, op.getType(), op.getA(),
                                                    op.getB());
    return success();
  }
};

/// Convert tosa.negate to tmatmul operations
/// Implements as: result = 0 - input
struct TosaNegateToTMatmulPattern : public OpRewritePattern<tosa::NegateOp> {
  using OpRewritePattern<tosa::NegateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::NegateOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getType();

    // Create zero constant (this would need proper constant handling)
    // For now, we'll need to handle this in a more sophisticated way
    // with proper constant folding

    return failure(); // TODO: Implement with constant support
  }
};

/// Pattern to decompose complex TOSA operations into simpler TMatmul ops
struct TosaCastToTMatmulPattern : public OpRewritePattern<tosa::CastOp> {
  using OpRewritePattern<tosa::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::CastOp op,
                                PatternRewriter &rewriter) const override {
    // For now, we'll elide casts that don't change the element type
    // More sophisticated handling would be needed for real cast operations
    auto inputType = op.getInput().getType().cast<ShapedType>();
    auto outputType = op.getType().cast<ShapedType>();

    if (inputType.getElementType() == outputType.getElementType()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct TosaToTMatmulPass
    : public PassWrapper<TosaToTMatmulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaToTMatmulPass)

  StringRef getArgument() const final { return "convert-tosa-to-tmatmul"; }

  StringRef getDescription() const final {
    return "Lower TOSA ops to TMatmul dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tmatmul::TMatmulDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);

    // Add conversion patterns
    populateTosaToTMatmulConversionPatterns(patterns);

    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::tosa::populateTosaToTMatmulConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TosaAddToTMatmulPattern, TosaSubToTMatmulPattern,
               TosaMulToTMatmulPattern, TosaSigmoidToTMatmulPattern,
               TosaFullyConnectedToTMatmulPattern, TosaMatMulToTMatmulPattern,
               TosaCastToTMatmulPattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::tosa::createTosaToTMatmulPass() {
  return std::make_unique<TosaToTMatmulPass>();
}
