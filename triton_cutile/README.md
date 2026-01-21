# Triton vs CuTile: Proving TileIR Advantages

A research project demonstrating the 4 code generation challenges that limit Triton on Hopper/Blackwell GPUs, and how CuTile/TileIR can overcome them.

## The Thesis

Modern GPUs (Hopper, Blackwell) have specialized fixed-function units (Tensor Cores, TMA) that require **cooperative multi-warp execution** and **async operations**. Traditional GPU compilers like Triton are built on **sequential IRs** (LLVM/PTX) and struggle with:

| Challenge | Problem | Triton Impact | CuTile Solution |
|-----------|---------|---------------|-----------------|
| **1. Cooperative Warps** | Large TC ops need multi-warp cooperation | Single-threaded SIMT model | Tile-level abstractions |
| **2. Register Pressure** | Modulo schedules + large TCs → >255 regs | Spilling penalty | Explicit memory management |
| **3. Variable Latency** | TMA transfers have 10x+ variance | Static sync fails | Dynamic async handling |
| **4. Blocking Sync** | `GEMM.WAIT` blocks concurrent EXP | Wasted functional units | Multi-warp distribution |

## Experiments

### 04_blocking_sync: GEMM + EXP Fusion (Challenge 4)

**Goal**: Show that Triton's same-warp execution forces `GEMM.WAIT` to block independent `EXP` operations, while CuTile can potentially overlap them.

```
┌─────────────────────────────────────────────────────────────────┐
│  The Paper's Figure 2 Scenario:                                 │
│                                                                 │
│  Single Warp (Triton):      Multi-Warp (CuTile potential):      │
│  ├─ GEMM (Tensor Core)      ├─ Warp 0: GEMM (Tensor Core)       │
│  ├─ WAIT ◀── blocks!        ├─ Warp 1: EXP (independent)        │
│  └─ EXP                     └─ Sync only when needed            │
└─────────────────────────────────────────────────────────────────┘
```

**Run locally (compile only)**:
```bash
python experiments/04_blocking_sync/triton_gemm_exp.py --compile-only
python experiments/04_blocking_sync/cutile_gemm_exp.py --compile-only
```

**Run on Modal B200**:
```bash
modal run run_modal.py --experiment 04_blocking_sync
```

## Project Structure

```
triton_cutile/
├── README.md                           # This file
├── run_modal.py                        # Modal deployment for B200
├── experiments/
│   └── 04_blocking_sync/               # Challenge 4 experiment
│       ├── triton_gemm_exp.py          # Triton: C = exp(A @ B)
│       ├── cutile_gemm_exp.py          # CuTile: C = exp(A @ B)
│       └── analyze.py                  # SASS comparison script
├── outputs/                            # Generated artifacts
│   ├── triton/                         # PTX, SASS, IR
│   └── cutile/                         # PTX, SASS, IR
└── docs/
    └── notes.md                        # Research notes
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton 3.0+
- cuda-tile (CuTile)
- Modal account (for B200 access)

## References

- Paper: "CuTile: Tile-Level Programming for GPUs" (discussing these 4 challenges)
- [Triton Documentation](https://triton-lang.org/)
- [cuda-tile Package](https://pypi.org/project/cuda-tile/)
