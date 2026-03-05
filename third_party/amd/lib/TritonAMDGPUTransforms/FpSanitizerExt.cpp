#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace amdgpu = mlir::triton::amdgpu;

#define GEN_PASS_DEF_TRITONAMDGPUFPSANITIZEREXT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

Value decodeE8M0ToFloat(PatternRewriter &rewriter, Location loc, Value scale,
                        FloatType dstElemTy) {
  auto scaleTy = cast<RankedTensorType>(scale.getType());
  // f16 can't represent E8M0's range. Use f32 to decode then cast.
  FloatType decodeTy = dstElemTy.isF16() ? rewriter.getF32Type() : dstElemTy;
  int width = decodeTy.getIntOrFloatBitWidth();
  int shift = decodeTy.getFPMantissaWidth() - 1;
  auto intTy = rewriter.getIntegerType(width);
  auto intTensorTy = scaleTy.clone(intTy);

  auto zext = arith::ExtUIOp::create(rewriter, loc, intTensorTy, scale);
  auto shiftConst =
      arith::ConstantIntOp::create(rewriter, loc, shift, width);
  auto shiftVec =
      tt::SplatOp::create(rewriter, loc, intTensorTy, shiftConst);
  auto shl = arith::ShLIOp::create(rewriter, loc, zext, shiftVec);
  Value result =
      tt::BitcastOp::create(rewriter, loc, scaleTy.clone(decodeTy), shl);
  if (decodeTy != dstElemTy)
    result = tt::FpToFpOp::create(rewriter, loc, scaleTy.clone(dstElemTy),
                                  result);
  return result;
}

Value convertScaleElemType(PatternRewriter &rewriter, Location loc, Value scale,
                           FloatType dstElemTy) {
  auto scaleTy = cast<RankedTensorType>(scale.getType());
  auto elemTy = scaleTy.getElementType();
  if (elemTy == dstElemTy)
    return scale;
  if (isa<FloatType>(elemTy))
    return tt::FpToFpOp::create(rewriter, loc, scaleTy.clone(dstElemTy),
                                scale);
  if (isa<IntegerType>(elemTy))
    return decodeE8M0ToFloat(rewriter, loc, scale, dstElemTy);
  return {};
}

Value
convertScaleToType(PatternRewriter &rewriter, Location loc,
                  Value scale, RankedTensorType dstTy) {
  auto dstElemTy = cast<FloatType>(dstTy.getElementType());
  auto scaled = convertScaleElemType(rewriter, loc, scale, dstElemTy);
  if (!scaled)
    return {};
  if (scaled.getType() != dstTy)
    scaled = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, scaled);
  return scaled;
}

//----------------------------------------
// Patterns
//----------------------------------------

struct ScaledUpcastFp8OpPattern :
    public OpRewritePattern<amdgpu::ScaledUpcastFp8Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(amdgpu::ScaledUpcastFp8Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstTy = op.getOutput().getType();
    auto dstElemTy = dstTy.getElementType();

    Value upcasted = tt::FpToFpOp::create(
        rewriter, loc, op.getInput().getType().clone(dstElemTy), op.getInput());

    auto scale = convertScaleToType(rewriter, loc, op.getScale(), dstTy);
    if (!scale)
      return failure();

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, upcasted, scale);
    return success();
  }
};

struct ScaledUpcastFp4OpPattern :
    public OpRewritePattern<amdgpu::ScaledUpcastFp4Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(amdgpu::ScaledUpcastFp4Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstTy = op.getOutput().getType();
    auto dstElemTy = dstTy.getElementType();
    Type upcastElemTy = dstElemTy.isF32() ? rewriter.getBF16Type() : dstElemTy;

    Value upcasted = ttg::Fp4ToFpOp::create(
        rewriter, loc, op.getInput(), upcastElemTy, op.getAxis());
    if (upcastElemTy != dstElemTy) {
      auto ty = cast<RankedTensorType>(upcasted.getType()).clone(dstElemTy);
      upcasted = tt::FpToFpOp::create(rewriter, loc, ty, upcasted);
    }
    // ScaledUpcastFp4Op does not have SameOperandsAndResultEncoding, so
    // maybe convert_layout to dstTy.
    if (upcasted.getType() != dstTy)
      upcasted = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, upcasted);

    auto scale = convertScaleToType(rewriter, loc, op.getScale(), dstTy);
    if (!scale)
      return failure();

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, upcasted, scale);
    return success();
  }
};

void populateAmdFpSanPatterns(RewritePatternSet &patterns) {
  patterns.add<ScaledUpcastFp4OpPattern,
               ScaledUpcastFp8OpPattern>(patterns.getContext());
}

class TritonAMDGPUFpSanitizerExtPass
    : public impl::TritonAMDGPUFpSanitizerExtBase<
          TritonAMDGPUFpSanitizerExtPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdFpSanPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation()->emitError(
          "FpSanitizer error: Failed to apply AMD patterns");
      signalPassFailure();
      return;
    }

    bool hasUnsupported = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<amdgpu::ScaledUpcastFp8Op, amdgpu::ScaledUpcastFp4Op>(op)) {
        op->emitError("FpSanitizer error: unsupported AMD op remaining: ")
            << op->getName();
        hasUnsupported = true;
      }
    });
    if (hasUnsupported)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
