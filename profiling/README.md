# Nsight traces and harness logs

Every folder corresponds to one experiment in the report. Each contains:

- `*_m3.out` / `m3_*.out` — raw stdout of the benchmark harness (per-layer time, total run
  time, test accuracy) captured while the profile was recorded;
- `*.ncu-rep` — Nsight Compute report for the kernel(s) of that variant;
- `req_0/req_0.nsys-rep` — Nsight Systems timeline used for the stream-overlap analysis.

| Folder | Variant | Batch-10k conv-1 / conv-2 | Report section |
|---|---|---:|---|
| `req_0/` | 3 CUDA streams + pinned memory | 481.95 / 366.06 ms | §1 |
| `req_1/` | Tensor Cores, TF32 WMMA 16×16×8 | 46.09 / 31.37 ms | §2 |
| `op_0/` | weights in `__constant__` memory | 34.29 / 48.54 ms | §3 |
| `op_1/` | `__restrict__` pointers | 34.25 / 30.11 ms | §4 |
| `op_2/` | manual loop unroll (`k += 4`) | 33.97 / 30.10 ms | §5 |
| `op_4/` | cuBLAS `cublasSgemm` | 97.30 / 62.32 ms | §6 |
| `op_5/` | FP16 `__half2` / `__hfma2` | 49.29 / 30.38 ms | §7 |

## Opening the reports

```bash
# Nsight Compute — GUI
ncu-ui profiling/op_2/loopunrolling_analysis_file.ncu-rep

# Nsight Compute — CLI summary
ncu --import profiling/op_2/loopunrolling_analysis_file.ncu-rep --page details

# Nsight Systems
nsys-ui profiling/req_0/req_0.nsys-rep
```

The `.ncu-rep` and `.nsys-rep` files are stored with Git LFS, so run `git lfs pull` after
cloning if your client did not fetch them automatically.

## Which file answers which question

- *Did the stream pipeline actually overlap copy and compute?* → `req_0/req_0.nsys-rep`
  (look at the CUDA HW "Memory" row against the kernel row).
- *Is the tensor pipe doing anything?* → `req_1/tensorcore_analysis_file.ncu-rep`
  (Details → GPU Speed Of Light → Pipe Utilization).
- *Did constant memory reduce global traffic?* → `op_0/constantweight_analysis_file.ncu-rep`
  (Memory Throughput Breakdown: L2 tag requests / sectors / fills).
- *How good are the caches after `__restrict__`?* →
  `op_1/restrict_analysis_file.ncu-rep` (L1/L2 hit rate, occupancy).
- *Why is 78 % SM utilization not a speedup?* →
  `op_2/loopunrolling_analysis_file.ncu-rep` (SOL, arithmetic intensity, occupancy).
- *Why is cuBLAS slower than the hand-written kernel?* →
  `op_4/cublas_analysis_file.ncu-rep` (93.9 % memory throughput, 22.9 % compute).
- *Is FP16 doing packed math?* → `op_5/fp16_analysis_file.ncu-rep` (FMA (FP16) pipe).
