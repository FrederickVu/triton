#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCOALESCEATOMIC
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

constexpr unsigned kMaxAtomicVecBits = 32;

// We cap the vectorization of atomic rmw ops to 32 bits for sub 64-bit dtypes
// as there are currently no instructions which support larger vectorization.
struct CoalesceAtomicPass
    : impl::TritonAMDGPUCoalesceAtomicBase<CoalesceAtomicPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);

    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](triton::AtomicRMWOp atomicOp) {
      Value ptr = atomicOp.getPtr();
      auto tensorType = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorType || !isa<PointerType>(tensorType.getElementType()))
        return;

      int numWarps = lookupNumWarps(atomicOp);
      auto cgaLayout = getCGALayout(tensorType.getEncoding());
      auto shapePerCTA = getShapePerCTA(tensorType);

      auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
      SmallVector<unsigned> order = getOrderFromContiguity(contiguity);

      unsigned perThread = getNumElementsPerThread(
          atomicOp, order, axisInfoAnalysis, shapePerCTA);

      unsigned elemBitWidth = getElementBitWidth(tensorType);
      unsigned maxPerThread = std::max(1u, kMaxAtomicVecBits / elemBitWidth);
      perThread = std::min<unsigned>(perThread, maxPerThread);

      SmallVector<unsigned> sizePerThread(tensorType.getRank(), 1);
      sizePerThread[order[0]] = perThread;

      auto newLayout = BlockedEncodingAttr::get(
          atomicOp.getContext(), tensorType.getShape(), sizePerThread, order,
          numWarps, threadsPerWarp, cgaLayout);
      layoutMap[atomicOp] = newLayout;
    });

    for (auto &[op, layout] : layoutMap) {
      convertDistributedOpEncoding(layout, op);
    }
  }
};

} // namespace
} // namespace mlir
