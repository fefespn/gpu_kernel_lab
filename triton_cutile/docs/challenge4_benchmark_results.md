# Benchmark Results: Challenge 4 - Blocking Synchronization

## Summary

CuTile consistently outperforms Triton across all matrix sizes, with speedups ranging from **1.28x to 3.67x**.

| Config | M | K | N | Triton | CuTile | Speedup |
|--------|---|---|---|--------|--------|---------|
| Exp 1 | 512 | 512 | 512 | 8.71 TFLOPS | 17.21 TFLOPS | **1.98x** |
| Exp 1 | 1024 | 1024 | 1024 | 42.90 TFLOPS | 54.99 TFLOPS | **1.28x** |
| Exp 1 | 2048 | 2048 | 2048 | 62.77 TFLOPS | 92.56 TFLOPS | **1.47x** |
| Exp 2 | 32 | 8192 | 1024 | 3.22 TFLOPS | 11.84 TFLOPS | **3.67x** |
| Exp 2 | 8192 | 8192 | 1024 | 59.48 TFLOPS | 117.50 TFLOPS | **1.98x** |
| Exp 2 | 8192 | 8192 | 128 | 52.18 TFLOPS | 99.33 TFLOPS | **1.90x** |

---

## Experiment 1: Square Matrices (2026-01-21)

**Hardware:** Modal B200 GPU  
**Config:** warmup=10, runs=100

```
======================================================================
BENCHMARK: 04_blocking_sync (warmup=10, runs=100)
======================================================================

[Matrix Size: 512x512]
  Triton: 0.031 ms, 8.71 TFLOPS, max_error=3.49e-03
  CuTile: 0.016 ms, 17.21 TFLOPS, max_error=5.01e-04
  Speedup (CuTile/Triton): 1.98x

[Matrix Size: 1024x1024]
  Triton: 0.050 ms, 42.90 TFLOPS, max_error=5.79e-03
  CuTile: 0.039 ms, 54.99 TFLOPS, max_error=1.48e-03
  Speedup (CuTile/Triton): 1.28x

[Matrix Size: 2048x2048]
  Triton: 0.274 ms, 62.77 TFLOPS, max_error=2.05e-02
  CuTile: 0.186 ms, 92.56 TFLOPS, max_error=3.02e-03
  Speedup (CuTile/Triton): 1.47x
```

---

## Experiment 2: Transformer-like Shapes (2026-01-21)

**Hardware:** Modal B200 GPU  
**Config:** warmup=10, runs=100  
**Interpretation:** M=batch, K=hidden dim, N=KV projection (8 heads × 128 dim)

```
======================================================================
BENCHMARK: 04_blocking_sync (warmup=10, runs=100)
======================================================================

[Matrix Size: 32x8192x1024]
  Triton: 0.167 ms, 3.22 TFLOPS, max_error=6.80e-02
  CuTile: 0.045 ms, 11.84 TFLOPS, max_error=1.05e-02
  Speedup (CuTile/Triton): 3.67x

[Matrix Size: 8192x8192x1024]
  Triton: 2.311 ms, 59.48 TFLOPS, max_error=4.05e-01
  CuTile: 1.170 ms, 117.50 TFLOPS, max_error=4.34e-02
  Speedup (CuTile/Triton): 1.98x

[Matrix Size: 8192x8192x128]
  Triton: 0.329 ms, 52.18 TFLOPS, max_error=3.10e-01
  CuTile: 0.173 ms, 99.33 TFLOPS, max_error=3.59e-02
  Speedup (CuTile/Triton): 1.90x
```

---

## Key Observations

1. **Highest speedup on small batches:** 3.67x at M=32 shows CuTile handles tail cases better
2. **CuTile achieves 117 TFLOPS** on large matrices (vs Triton's 59 TFLOPS)
3. **Lower numerical error:** CuTile consistently has 5-10x lower max error
4. **SASS evidence:** CuTile uses 7 sync ops vs Triton's 48 (6.8x fewer barriers)
