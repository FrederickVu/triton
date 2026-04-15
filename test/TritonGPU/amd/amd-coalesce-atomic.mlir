// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-atomic | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: atomic_rmw_bf16
tt.func @atomic_rmw_bf16(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
  %1 = tt.splat %base : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
  %ptr = tt.addptr %1, %0 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
  %val = arith.constant dense<1.0> : tensor<1024xbf16, #blocked>
  // CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}} : (tensor<1024x!tt.ptr<bf16>, #[[$COALESCED]]>
  %2 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: atomic_rmw_f32
tt.func @atomic_rmw_f32(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
  %1 = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
  %ptr = tt.addptr %1, %0 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
  %val = arith.constant dense<1.0> : tensor<256xf32, #blocked>
  // CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}} : (tensor<256x!tt.ptr<f32>, #[[$COALESCED]]>
  %2 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: atomic_rmw_f64
tt.func @atomic_rmw_f64(%base: !tt.ptr<f64> {tt.divisibility = 16 : i32}) {
  %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
  %1 = tt.splat %base : !tt.ptr<f64> -> tensor<256x!tt.ptr<f64>, #blocked>
  %ptr = tt.addptr %1, %0 : tensor<256x!tt.ptr<f64>, #blocked>, tensor<256xi32, #blocked>
  %val = arith.constant dense<1.0> : tensor<256xf64, #blocked>
  // CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}} : (tensor<256x!tt.ptr<f64>, #[[$COALESCED]]>
  %2 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<256x!tt.ptr<f64>, #blocked>, tensor<256xf64, #blocked>) -> tensor<256xf64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice0 = #ttg.slice<{dim = 1, parent = #blocked}>
#slice1 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: atomic_rmw_bf16_2d
tt.func @atomic_rmw_bf16_2d(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %stride: i32 {tt.divisibility = 16 : i32}) {
  %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice0>
  %c = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice1>
  %r2d = tt.expand_dims %r {axis = 1 : i32} : tensor<128xi32, #slice0> -> tensor<128x1xi32, #blocked>
  %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32, #slice1> -> tensor<1x32xi32, #blocked>
  %stride_splat = tt.splat %stride : i32 -> tensor<1x32xi32, #blocked>
  %col_offset = arith.muli %c2d, %stride_splat : tensor<1x32xi32, #blocked>
  %r_bc = tt.broadcast %r2d : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
  %c_bc = tt.broadcast %col_offset : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
  %offsets = arith.addi %r_bc, %c_bc : tensor<128x32xi32, #blocked>
  %base_splat = tt.splat %base : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, #blocked>
  %ptr = tt.addptr %base_splat, %offsets : tensor<128x32x!tt.ptr<bf16>, #blocked>, tensor<128x32xi32, #blocked>
  %val = arith.constant dense<1.0> : tensor<128x32xbf16, #blocked>
  // CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %{{.*}} : (tensor<128x32x!tt.ptr<bf16>, #[[$COALESCED]]>
  %0 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<128x32x!tt.ptr<bf16>, #blocked>, tensor<128x32xbf16, #blocked>) -> tensor<128x32xbf16, #blocked>
  tt.return
}
}
