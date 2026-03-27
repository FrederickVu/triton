/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEEPILOGUE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

bool isOneOperandElementwiseOp(Operation *op) {
  if (llvm::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::NegFOp, arith::SIToFPOp,
                arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::LogOp, math::Log10Op,
                math::Log1pOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
                math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
           externElementwiseOp.getPure();
  return false;
}

// Tries to optimize oldStoreOp with v_permlane*_swap instruction when possible.
// Returns null store op if not suitable.
static triton::StoreOp
usePermlaneSwapToOptimizeStore(PatternRewriter &rewriter, Value ptr, Value val,
                               Value mask, triton::StoreOp oldStoreOp) {
  auto ptrType = cast<RankedTensorType>(ptr.getType());
  auto valType = cast<RankedTensorType>(val.getType());

  // Create a new layout where each thread holds 8 consecutive elements, in
  // order to enable wide 128-bit global stores.
  std::optional<triton::LinearLayout> storeLL =
      triton::gpu::chooseMfmaLikeStoreLayout(valType);
  if (!storeLL)
    return nullptr;

  Attribute newEncoding = triton::gpu::LinearEncodingAttr::get(
      oldStoreOp.getContext(), std::move(storeLL.value()));
  auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
  Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                      newPtrType, ptr);

  auto newValType = valType.cloneWithEncoding(newEncoding);
  Value newVal = triton::gpu::ConvertLayoutOp::create(rewriter, val.getLoc(),
                                                      newValType, val);

  Value newMask = mask;
  if (mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto newMaskType = maskType.cloneWithEncoding(newEncoding);
    newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                   newMaskType, mask);
  }

  return triton::StoreOp::create(rewriter, oldStoreOp.getLoc(), newPtr, newVal,
                                 newMask, oldStoreOp.getCache(),
                                 oldStoreOp.getEvict());
}

// convert(val) : xmma -> blocked
// elementWiseOp(val) : blocked
// ...
// elementWiseOp(val) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val) : xmma
// ...
// elementWiseOp(val) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA
class BypassEpilogueSMEM : public mlir::OpRewritePattern<triton::StoreOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp stOp,
                  mlir::PatternRewriter &rewriter) const override {

    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> chainedOps;
    while (true) {
      auto chainedOp = val.getDefiningOp();
      if (!chainedOp)
        return mlir::failure();
      if (llvm::isa<triton::gpu::ConvertLayoutOp>(chainedOp))
        break;
      if (!chainedOp->hasOneUse())
        return mlir::failure();
      if (!isOneOperandElementwiseOp(chainedOp))
        return mlir::failure();
      val = chainedOp->getOperand(0);
      chainedOps.push_back(chainedOp);
    }

    auto cvtOp = val.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();

    auto encoding = cvtOp.getSrc().getType().getEncoding();
    if (!isa<triton::gpu::MmaEncodingTrait>(encoding))
      return mlir::failure();

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
    Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                        newPtrType, ptr);

    auto newVal = cvtOp.getSrc();

    for (auto chainedOp : llvm::reverse(chainedOps)) {
      auto oldType =
          cast<mlir::RankedTensorType>(chainedOp->getResult(0).getType());
      chainedOp->setOperand(0, newVal);
      newVal = llvm::cast<mlir::TypedValue<RankedTensorType>>(
          chainedOp->getResult(0));

      auto newType = oldType.cloneWithEncoding(newEncoding);
      newVal.setType(newType);
    }

    Value newMask = mask;
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskType = maskType.cloneWithEncoding(newEncoding);
      newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                     newMaskType, mask);
    }
    triton::StoreOp newStoreOp =
        usePermlaneSwapToOptimizeStore(rewriter, newPtr, newVal, newMask, stOp);
    if (!newStoreOp) {
      newStoreOp =
          triton::StoreOp::create(rewriter, stOp.getLoc(), newPtr, newVal,
                                  newMask, stOp.getCache(), stOp.getEvict());
    }

    rewriter.replaceOp(stOp, newStoreOp);
    return mlir::success();
  }
};

bool hasTransposedXmmaLayout(RankedTensorType ty) {
  auto enc = ty.getEncoding();
  bool isTransposed = false;
  if (auto mfmaEnc = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(enc))
    isTransposed = mfmaEnc.getIsTransposed();
  else if (auto wmmaEnc = dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(enc))
    isTransposed = wmmaEnc.getIsTransposed();

  return isTransposed;
}

std::optional<triton::LinearLayout>
chooseNewAtomicSinkLayout(RankedTensorType srcTy,
                          triton::AxisInfo *ptrAxisInfo) {
  auto contig = ptrAxisInfo->getContiguity();
  auto order = getOrderFromContiguity(contig);
  if (order[0] != 0 || !hasTransposedXmmaLayout(srcTy))
    return std::nullopt;

  auto *ctx = srcTy.getContext();
  auto srcLL = triton::gpu::toLinearLayout(srcTy);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto newBases = srcLL.getBases();
  std::swap(newBases[kReg][0], newBases[kLane][0]);
  auto outDimNames = llvm::to_vector(srcLL.getOutDimNames());
  return triton::LinearLayout(std::move(newBases), outDimNames);
}

void relayoutAtomicChain(triton::AtomicRMWOp atomicOp,
                         triton::gpu::ConvertLayoutOp cvtOp,
                         ArrayRef<Operation *> chainedOps, Attribute newEnc,
                         PatternRewriter &rewriter) {
  auto loc = atomicOp.getLoc();
  cvtOp.getResult().setType(
      cast<RankedTensorType>(cvtOp.getResult().getType())
          .cloneWithEncoding(newEnc));

  for (auto *op : llvm::reverse(chainedOps)) {
    auto oldTy = cast<RankedTensorType>(op->getResult(0).getType());
    op->getResult(0).setType(oldTy.cloneWithEncoding(newEnc));
  }

  auto ptrTy = cast<RankedTensorType>(atomicOp.getPtr().getType());
  Value newPtr = triton::gpu::ConvertLayoutOp::create(
      rewriter, loc, ptrTy.cloneWithEncoding(newEnc), atomicOp.getPtr());
  atomicOp.getPtrMutable().assign(newPtr);

  if (atomicOp.getMask()) {
    auto maskTy = cast<RankedTensorType>(atomicOp.getMask().getType());
    Value newMask = triton::gpu::ConvertLayoutOp::create(
        rewriter, loc, maskTy.cloneWithEncoding(newEnc), atomicOp.getMask());
    atomicOp.getMaskMutable().assign(newMask);
  }

  atomicOp.getResult().setType(
      cast<RankedTensorType>(atomicOp.getResult().getType())
          .cloneWithEncoding(newEnc));
}

class RelayoutAtomicSink
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
public:
  RelayoutAtomicSink(MLIRContext *context,
                     triton::AMD::ModuleAxisInfoAnalysis &axisInfo)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        axisInfo(axisInfo) {}

  // If bf16/fp16 atomic add is a sink and is downstream of mfma/wmma encoding
  // along single-use chain of layout-preserving ops, and the mma encoding
  // lacks vectorization along contiguous dimension, then rewrite chain and
  // atomic with vectorized encoding using a cheap ConvertLayoutOp.
  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getResult().use_empty())
      return mlir::failure();
    if (op.getAtomicRmwOp() != triton::RMWOp::FADD)
      return mlir::failure();

    auto elemTy = getElementTypeOrSelf(op.getVal());
    if (!elemTy.isF16() && !elemTy.isBF16())
      return mlir::failure();

    SmallVector<Operation *> chainedOps;
    auto root = peelOneUseUnaryElementwiseOps(op.getVal(), chainedOps);
    auto cvtOp = root.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp || !cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto *ptrAxisInfo = axisInfo.getAxisInfo(op.getPtr());
    if (!ptrAxisInfo)
      return mlir::failure();
    auto newLL = chooseNewAtomicSinkLayout(
        cast<RankedTensorType>(cvtOp.getSrc().getType()), ptrAxisInfo);
    if (!newLL)
      return mlir::failure();

    auto newEnc = triton::gpu::LinearEncodingAttr::get(op->getContext(),
                                                       std::move(*newLL));
    relayoutAtomicChain(op, cvtOp, chainedOps, newEnc, rewriter);
    return mlir::success();
  }

private:
  triton::AMD::ModuleAxisInfoAnalysis &axisInfo;
};

} // anonymous namespace

class TritonAMDGPUOptimizeEpiloguePass
    : public impl::TritonAMDGPUOptimizeEpilogueBase<
          TritonAMDGPUOptimizeEpiloguePass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    triton::AMD::ModuleAxisInfoAnalysis axisInfo(m);

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM>(context);
    patterns.add<RelayoutAtomicSink>(context, axisInfo);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
