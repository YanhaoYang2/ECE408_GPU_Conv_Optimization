# Kernel source

`conv_tensor_core_wmma.cu` is the Req_1 (Tensor Core) version of the convolution
implementation, kept verbatim from the project:

- `matrixMultiplyTF32` — GEMM mapped onto TF32 Tensor Cores with the CUDA `wmma` API
  (`16×16×8` fragments, one warp per block, shared-memory staging, zero-padded boundary
  tiles);
- `matrix_unrolling_kernel` — the im2col/unroll kernel used by the non-fused path;
- `matrix_permute_kernel` — converts `Map_out × Batch × H × W` into the `NCHW` output layout;
- `GPUInterface::conv_forward_gpu_*` — host-side prologue/epilogue.

## Provenance and buildability

This file was written against the **ECE 408 MP3 project skeleton**, which is course-provided
and is *not* redistributed here. It therefore references two headers that are not in this
repository:

```cpp
#include "gpu-new-forward.h"   // course skeleton: GPUInterface declaration
#include "matmul.h"            // course skeleton: MATMUL_TILE_WIDTH etc.
```

As a result the file is a **readable reference implementation, not a standalone build**.
The kernel bodies and host code are unmodified, so the WMMA fragment setup, the K-dimension
sweep and the boundary handling can be compared directly against the report
([§2, Tensor Cores](../docs/report/ECE408_M3_GEMM_Optimization_Report.pdf)).

If you want to compile it, drop it into an MP3 skeleton and provide those two headers; the
Makefile in the skeleton already links `-lcublas -lcudart`.

## Notes on the code

- Only one warp (`threadIdx.x < 32`) runs the WMMA section, which is exactly why the report
  measures ~4 active warps/scheduler and a tensor-pipe utilization of ~4.6 % — the kernel is
  correct but occupancy-limited.
- `A_tile` / `B_tile` are reloaded per K-tile without double buffering, so the single warp
  stalls on global loads; a production version would use multiple warps, `cp.async` and
  `m16n16k16` FP16 MMA.
- The `if (tx == 0)` clamp after `store_matrix_sync` serialises the epilogue; it is a
  correctness-first choice for boundary tiles.
