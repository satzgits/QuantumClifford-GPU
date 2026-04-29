# Proof of Concept: QuantumClifford.jl CPU Baseline Benchmark
# GSoC 2026 Proposal - GPU-Accelerated Simulator of Clifford Circuits
#
# This script establishes baseline CPU performance for GPU comparison.
# Run this script to measure current CPU performance before GPU acceleration.
#
# Usage:
#   julia poc_quantumclifford.jl
#
# Requirements:
#   - Julia 1.10+
#   - QuantumClifford.jl
#   - BenchmarkTools.jl
#
# Installation (run in Julia REPL):
#   using Pkg
#   Pkg.add("QuantumClifford")
#   Pkg.add("BenchmarkTools")

println("="^70)
println("QuantumClifford.jl CPU Baseline Benchmark")
println("GSoC 2026 Proof of Concept - GPU Acceleration for Clifford Circuits")
println("="^70)

# Check if packages are installed, install if needed
function ensure_packages()
    packages = ["QuantumClifford", "BenchmarkTools", "Printf"]
    for pkg in packages
        try
            @eval using $(Symbol(pkg))
        catch e
            println("Installing $pkg...")
            import Pkg
            Pkg.add(pkg)
        end
    end
end

ensure_packages()

using QuantumClifford
using BenchmarkTools
using Printf

# Print version information
println("\n" * "="^70)
println("Environment Information")
println("="^70)
@printf "Julia Version: %s\n" VERSION
@printf "QuantumClifford Version: %s\n" pkgversion(QuantumClifford)

# Get system info
println("\nSystem Info:")
println("  Threads: $(Threads.nthreads())")
println("  CPU: $(Sys.CPU_INFO[1].vendor])")

# ============================================================================
# Test 1: Pauli-Pauli Multiplication
# ============================================================================
println("\n" * "="^70)
println("Test 1: Pauli-Pauli Multiplication")
println("="^70)
println("\nDescription: Multiply two random Pauli operators")
println("Expected CPU time: ~13ms for 10000 qubits (from library benchmarks)")
println("-"^70)

for n_qubits in [100, 1000, 10000]
    println("\n  Qubits: $n_qubits")
    p1 = random_pauli(n_qubits)
    p2 = random_pauli(n_qubits)
    
    # Warm-up
    p_result = p1 * p2
    
    # Benchmark
    bench = @benchmark $p1 * $p2 samples=100 evals=1
    
    @printf "  Median time: %.3f ms\n" minimum(bench.times) / 1e6
    @printf "  Mean time:   %.3f ms\n" mean(bench.times) / 1e6
    @printf "  Memory:      %.2f KiB\n" bench.memory / 1024
    @printf "  Allocations: %d\n" bench.allocs
end

# ============================================================================
# Test 2: Stabilizer State Creation and Canonicalization
# ============================================================================
println("\n" * "="^70)
println("Test 2: GHZ State Creation and Canonicalization")
println("="^70)
println("\nDescription: Create GHZ state and canonicalize the stabilizer")
println("Expected CPU time: ~9ms for 1000 qubits (from library benchmarks)")
println("-"^70)

for n_qubits in [100, 500, 1000]
    println("\n  Qubits: $n_qubits")
    ghz_state = stabilizerGHZ(n_qubits)
    
    # Warm-up
    canonicalize!(ghz_state)
    
    # Benchmark
    bench = @benchmark canonicalize!($ghz_state) samples=100 evals=1
    
    @printf "  Median time: %.3f ms\n" minimum(bench.times) / 1e6
    @printf "  Mean time:   %.3f ms\n" mean(bench.times) / 1e6
    @printf "  Memory:      %.2f KiB\n" bench.memory / 1024
    @printf "  Allocations: %d\n" bench.allocs
end

# ============================================================================
# Test 3: Clifford Gate Application
# ============================================================================
println("\n" * "="^70)
println("Test 3: Clifford Gate Application")
println("="^70)
println("\nDescription: Apply CNOT gate to stabilizer state")
println("Expected CPU time: ~3μs for sparse gate on 1000 qubits")
println("-"^70)

for n_qubits in [100, 500, 1000]
    println("\n  Qubits: $n_qubits")
    state = random_stabilizer(n_qubits)
    
    # Warm-up
    apply!(state, tCNOT)
    
    # Benchmark
    bench = @benchmark apply!($state, tCNOT) samples=1000 evals=10
    
    @printf "  Median time: %.3f μs\n" minimum(bench.times) / 1e3
    @printf "  Mean time:   %.3f μs\n" mean(bench.times) / 1e3
    @printf "  Memory:      %.2f bytes\n" bench.memory
    @printf "  Allocations: %d\n" bench.allocs
end

# ============================================================================
# Test 4: Dense Tableau Multiplication
# ============================================================================
println("\n" * "="^70)
println("Test 4: Dense Clifford Operator Multiplication")
println("="^70)
println("\nDescription: Multiply two dense Clifford operators")
println("Expected CPU time: ~17ms for 500 CNOTs on 1000 qubits")
println("-"^70)

for n_qubits in [50, 100, 200]
    println("\n  Qubits: $n_qubits")
    c1 = random_clifford(n_qubits)
    c2 = random_clifford(n_qubits)
    
    # Warm-up
    c_result = c1 * c2
    
    # Benchmark
    bench = @benchmark $c1 * $c2 samples=50 evals=1
    
    @printf "  Median time: %.3f ms\n" minimum(bench.times) / 1e6
    @printf "  Mean time:   %.3f ms\n" mean(bench.times) / 1e6
    @printf "  Memory:      %.2f KiB\n" bench.memory / 1024
    @printf "  Allocations: %d\n" bench.allocs
end

# ============================================================================
# Test 5: MixedDestabilizer Operations
# ============================================================================
println("\n" * "="^70)
println("Test 5: MixedDestabilizer Operations")
println("="^70)
println("\nDescription: Create and manipulate mixed stabilizer states")
println("-"^70)

for n_qubits in [50, 100, 200]
    println("\n  Qubits: $n_qubits")
    mixed_state = random_mixed_stabilizer(n_qubits)
    
    # Warm-up
    canonicalize!(mixed_state)
    
    # Benchmark
    bench = @benchmark canonicalize!($mixed_state) samples=50 evals=1
    
    @printf "  Median time: %.3f ms\n" minimum(bench.times) / 1e6
    @printf "  Mean time:   %.3f ms\n" mean(bench.times) / 1e6
    @printf "  Memory:      %.2f KiB\n" bench.memory / 1024
    @printf "  Allocations: %d\n" bench.allocs
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Benchmark Complete!")
println("="^70)
println("""
These CPU baseline measurements will be compared against GPU-accelerated
implementations using KernelAbstractions.jl. Expected GPU speedups:

  - Pauli-Pauli Products:     10-100× (for 1000+ qubits)
  - Clifford Gate Application: 2-5×   (for batch operations)
  - Tableau Multiplication:    5-20×  (for 500+ qubits)
  - Canonicalization:          2-10×  (with GPU Gaussian elimination)

Next Steps:
  1. Install GPU dependencies: CUDA.jl, KernelAbstractions.jl
  2. Run poc_gpu_kernel_sketch.jl for GPU implementation preview
  3. Compare GPU vs CPU performance

GitHub: https://github.com/QuantumSavory/QuantumClifford.jl
Proposal: See cvprop.md for detailed GSoC 2026 proposal
""")

println("="^70)
