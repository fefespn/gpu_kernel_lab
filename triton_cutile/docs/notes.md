# Research Notes: Challenge 4 - Blocking Synchronization

## The Problem (from the paper)

> "Finally, the fixed-function units on Hopper and Blackwell have asynchronous interfaces 
> that require explicit blocking synchronization to consume results. Since GPU threads have 
> in-order instruction issue, blocking synchronization interrupts the instruction issue of 
> any concurrently scheduled operations on the same warp."

### Visual Representation

```
SINGLE WARP EXECUTION (Triton):
================================
Time →

Warp 0:
├── HMMA.xxx (Tensor Core) ─────────────┐
├── HMMA.xxx                            │
├── ...                                 │
├── DEPBAR.LE (wait for GEMM)  ◀────────┘  ← BLOCKS HERE!
├── MUFU.EX2 (exp)                         ← Can only start now
├── MUFU.EX2
└── ST (store)

MULTI-WARP POTENTIAL (TileIR/CuTile):
=====================================
Time →

Warp 0: HMMA ─────────────────────────────→
Warp 1: HMMA ─────────────────────────────→
...
Warp 4: MUFU.EX2 (can start independently!)
Warp 5: MUFU.EX2
```

## What to Look For in SASS

### Triton Pattern (blocking)
```sass
// K-loop GEMM
HMMA.16816.F32.TF32 ...
HMMA.16816.F32.TF32 ...
...
// Sync before EXP
DEPBAR.LE SB0, 0x0    ; or BAR.SYNC or other wait
// EXP after sync
MUFU.EX2 R4, R8       ; exp2(x)
MUFU.EX2 R5, R9
```

### CuTile Pattern (potentially optimized)
If TileIR distributes across warps:
```sass
// Warp 0
HMMA.16816.F32.TF32 ...
// Warp 1 (interleaved)
MUFU.EX2 R4, R8       ; working on previous tile
```

## Experiment Setup

### Kernel: C = exp(A @ B)
- Matrix size: 1024×1024
- Tile: 64×64×32
- fp32 inputs, TF32 compute

### Commands
```bash
# Compile locally (no GPU needed)
cd triton_cutile
python experiments/04_blocking_sync/triton_gemm_exp.py --compile-only
python experiments/04_blocking_sync/cutile_gemm_exp.py --compile-only
python experiments/04_blocking_sync/analyze.py

# Run on Modal B200
modal run run_modal.py --experiment 04_blocking_sync
```

## Expected Results

1. **Triton SASS**: Clear blocking pattern (HMMA → WAIT → MUFU)
2. **CuTile SASS**: Potentially fewer explicit waits (TileIR optimization)

## Open Questions

- Does CuTile actually distribute across warps for this pattern?
- Is the Tile IR modulo scheduler doing something smarter?
- What compiler flags affect this behavior?
