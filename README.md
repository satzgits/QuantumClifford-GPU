# Proof of Concept: GPU-Accelerated Kernels for QuantumClifford.jl

Supporting repository for GSoC 2026 application: **GPU-Accelerated Simulator of Clifford Circuits** (Issue #553).

Contains CPU vs GPU benchmarking frameworks, CUDA.jl kernel sketches, and technical implementation notes for stabilizer tableau acceleration.

## Files

| File | Description |
|------|-------------|
| `benchmark_suite.jl` | Comprehensive CPU/GPU benchmarking framework. Measures Pauli multiplication, canonicalization, gate application, and memory transfer overhead across 100–5000 qubits. |
| `canonicalization_ka.jl` | Working CUDA.jl reference implementations for vector XOR, matrix row XOR, and batch operations on GF(2) stabilizer tableaux. |
| `poc_gpu_kernel_sketch.jl` | Kernel sketches demonstrating proposed GPU kernel design for row elimination and bit-packed XOR. |
| `poc_quantumclifford.jl` | CPU baseline benchmarks establishing current `QuantumClifford.jl` performance metrics. |
| `notes.md` | Technical notes on KA.jl compatibility, phase tracking, and memory layout considerations. |

## Key Results

- **CPU Baseline:** Established performance scaling for `canonicalize!`, `mul_left!`, and `apply!` up to 5000 qubits.
- **GPU Reference:** Verified `CuArray` row-XOR kernel correctness against CPU baseline for packed GF(2) operations.
- **Memory Analysis:** Documented PCI-e transfer overheads and identified 500+ qubit break-even threshold for GPU offloading.

## Quick Start

```bash
# CPU baseline
julia poc_quantumclifford.jl

# Full benchmark suite
julia benchmark_suite.jl

# GPU kernel verification (requires CUDA.jl)
julia canonicalization_ka.jl
```

## Related
- **Issue #553:** https://github.com/QuantumSavory/QuantumClifford.jl/issues/553
- **GSoC 2026:** https://julialang.org/jsoc/gsoc/quantumclifford/
- **Main Proposal:** [Proposal_quantum_clifford.pdf](../Julia/Proposal_quantum_clifford.pdf)
