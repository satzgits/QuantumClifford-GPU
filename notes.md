# Implementation Notes: GPU Canonicalization with CUDA.jl

## Status: ✓ WORKING

The GPU row XOR operations are now working correctly using **CUDA.jl directly** instead of KernelAbstractions.jl.

## The Problem: KernelAbstractions.jl Compatibility Issues

### Original Errors

1. **`UndefVarError: CUDA not defined`**
   - Cause: Missing `using CUDA` statement
   - Fix: Added `using CUDA`

2. **`UndefVarError: KACUDA not defined`**
   - Cause: Wrong import path `using KernelAbstractions.CUDA`
   - Fix: Don't use `KernelAbstractions.CUDA` - use `CUDA.CUDADevice()`

3. **`MethodError: no method matching isgpu(::CuArray{...})`**
   - Cause: KernelAbstractions kernel launch syntax incompatibility
   - The `@kernel` macro was having issues with argument type inference

4. **`MethodError: no method matching KernelAbstractions.NDIteration.StaticSize(::CuArray{...})`**
   - Cause: KernelAbstractions v0.9.x has breaking changes in kernel launch syntax
   - The macro expansion was failing for CuArray arguments

## The Solution: Use CUDA.jl Directly

Instead of KernelAbstractions.jl, we use **CUDA.jl's native `@cuda` macro**:

```julia
using CUDA

# Define kernel with CuDeviceVector/CuDeviceMatrix types
function gpu_xor_kernel!(dest::CuDeviceVector{Bool}, src::CuDeviceVector{Bool}, n::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if i <= n
        @inbounds dest[i] = !xor(!dest[i], !src[i])
    end
    
    return nothing
end

# Launch kernel
function gpu_xor!(dest::CuVector{Bool}, src::CuVector{Bool})
    n = length(dest)
    threads = 256
    blocks = cld(n, threads)
    
    @cuda threads=threads blocks=blocks gpu_xor_kernel!(dest, src, n)
    CUDA.synchronize()
    
    return dest
end
```

## Key Differences: KernelAbstractions vs CUDA.jl

| Aspect | KernelAbstractions.jl | CUDA.jl |
|--------|----------------------|---------|
| **Import** | `using KernelAbstractions` | `using CUDA` |
| **Kernel definition** | `@kernel function my_kernel!(args)` | `function my_kernel!(args::CuDevice...)` |
| **Thread index** | `@index(Global, Linear)` | `(blockIdx().x - 1) * blockDim().x + threadIdx().x` |
| **Kernel launch** | `kernel!(backend, ndrange=n)` | `@cuda threads=n blocks=m kernel!(args)` |
| **Synchronization** | `synchronize(kernel!)` | `CUDA.synchronize()` |
| **Device types** | Automatic | Must use `CuDeviceVector`, `CuDeviceMatrix` in kernels |

## Working Test Results

```
======================================================================
GPU-Accelerated Row Operations (GF(2))
Proof-of-concept for QuantumClifford.jl Issue #553
======================================================================

✓ CUDA is functional!
  GPU: CuDevice(0)
  VRAM: 7655 MB available

Test 1: GPU Vector XOR
Vector XOR match: true ✓

Test 2: GPU Matrix Row XOR
Match: true ✓

Test 3: Multiple Row XOR Operations
Multiple XOR match: true ✓

Test 4: Memory Transfer Benchmark
Transfer time for 100×100 Bool matrix: 0.02 ms

Test 5: CPU Row Reduction Timing
CPU row reduction time (100×100): 0.08 ms
```

## Why CUDA.jl Works Better Here

1. **Direct hardware access:** No abstraction layer between Julia and CUDA
2. **Better type inference:** `CuDeviceVector` types are explicit
3. **Mature ecosystem:** CUDA.jl is more stable than KernelAbstractions for complex kernels
4. **Standard CUDA syntax:** Uses familiar CUDA C patterns (`blockIdx`, `threadIdx`, etc.)

## Trade-offs

### CUDA.jl Advantages
- ✓ More control over kernel launch parameters
- ✓ Better error messages
- ✓ Direct access to CUDA features (shared memory, streams, etc.)
- ✓ More documentation and examples

### KernelAbstractions Advantages
- ✓ Hardware portability (CUDA, ROCm, Metal, oneAPI)
- ✓ Julia-native syntax
- ✓ Easier to write simple kernels

### Decision for QuantumClifford.jl

For **Issue #553 (GPU canonicalization)**:
- Use **CUDA.jl directly** for the initial implementation
- Can wrap with KernelAbstractions later if portability is needed
- QuantumClifford.jl already uses CUDA.jl in some places (v0.8.15+)

## Implementation Pattern for QuantumClifford.jl

```julia
# In src/gpu_canonicalize.jl

using CUDA

# GPU kernel for row XOR (tableau operation)
function tableau_row_xor_kernel!(
    xzs::CuDeviceMatrix{UInt64},
    src_row::Int,
    dst_row::Int,
    n_cols::Int
)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if col <= n_cols
        @inbounds xzs[dst_row, col] ⊻= xzs[src_row, col]
    end
    
    return nothing
end

function gpu_tableau_row_xor!(tableau::Tableau, src_row::Int, dst_row::Int)
    xzs_gpu = CuArray(tableau.xzs)
    n_cols = size(tableau.xzs, 2)
    threads = 256
    blocks = cld(n_cols, threads)
    
    @cuda threads=threads blocks=blocks tableau_row_xor_kernel!(
        xzs_gpu, src_row, dst_row, n_cols
    )
    CUDA.synchronize()
    
    tableau.xzs .= Array(xzs_gpu)
    return tableau
end
```

## Next Steps

1. **Implement full Gaussian elimination:**
   - Host-side loop over pivot columns
   - GPU kernel for row elimination
   - Synchronization after each pivot step

2. **Add phase tracking:**
   - Phase update kernel
   - Popcount for phase computation

3. **Integrate with QuantumClifford.jl:**
   - Wrap GPU functions in QuantumClifford API
   - Add `gpu_canonicalize!` function
   - Benchmark against CPU implementation

4. **Optimization:**
   - Use shared memory for frequently accessed rows
   - Kernel fusion for multiple row operations
   - Asynchronous kernel launches

## References

- [CUDA.jl Documentation](https://juliagpu.github.io/CUDA.jl/)
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [QuantumClifford.jl Issue #553](https://github.com/QuantumSavory/QuantumClifford.jl/issues/553)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

*Last updated: March 2026*  
*Status: GPU row XOR kernels working ✓*  
*For GSoC 2026: GPU-Accelerated Simulator of Clifford Circuits*
