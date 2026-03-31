# QuantumClifford-GPU
GPU-accelerated Clifford circuit simulation using CUDA.jl for QuantumClifford.jl
# GSoC 2026: GPU-Accelerated Clifford Circuit Simulator

**Applicant:** Savith 

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
