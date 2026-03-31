<<<<<<< HEAD
# GSoC 2026: GPU-Accelerated Clifford Circuit Simulator

**Project:** QuantumClifford.jl — Julia Society / GSoC 2026

---

## Overview

Proof of concept for GPU-accelerated stabilizer tableau operations using CUDA.jl.

## Files

| File | Description |
|------|-------------|
| `canonicalization_ka.jl` | Working GPU kernels (vector XOR, row XOR) |
| `benchmark_simple.jl` | CPU benchmarks for QuantumClifford.jl |
| `notes.md` | Technical notes on implementation |

## Quick Start

```julia
julia canonicalization_ka.jl
```

Test Results
GPU-Accelerated Row Operations (GF(2))
✓ Vector XOR match: true
✓ Matrix Row XOR match: true
✓ Multiple XOR match: true

Related
Issue #553: https://github.com/QuantumSavory/QuantumClifford.jl/issues/553
GSoC 2026: https://julialang.org/jsoc/gsoc/quantumclifford/
For GSoC 2026 Application
