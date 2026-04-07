# NumPy Compatibility

A summary of what ArraySwift implements relative to NumPy, and known limitations.

## Overview

ArraySwift is inspired by NumPy but is not a drop-in replacement. This document lists what is implemented, what differs, and what is not yet available.

## Supported Data Types

| NumPy dtype | ArraySwift | Notes |
|---|---|---|
| `float64` | `ArrayDType.float64` | Default type |
| `complex128` | `ArrayDType.complex128` | Split storage (real/imag arrays) |
| `float32`, `int*`, `bool` | Not supported | Only float64 and complex128 |

## Implemented Features

### Array Creation
- `zeros`, `ones`, `full`, `empty` -- with dtype parameter
- `arange`, `linspace`, `logspace`, `geomspace`
- `eye`, `identity`, `diag` (with complex support)
- `random`, `randn`, `randint`
- `meshgrid` (with `indexing` parameter: "xy" and "ij")
- `zerosLike`, `onesLike`, `fullLike`, `emptyLike`

### Array Manipulation
- `reshape` (supports -1 inference)
- `flatten`, `ravel`
- `transpose`, `.T` shorthand
- `squeeze`, `expandDims`, `swapaxes`, `moveaxis`
- `concatenate`, `stack`, `split`
- `tile`, `repeat`, `flip`, `roll`

### Mathematical Functions
- Trigonometric: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`
- Exponential/Log: `exp`, `log`, `log2`, `log10`, `log1p`, `expm1`
- Other: `sqrt`, `square`, `abs`, `sign`, `clip`, `round`, `floor`, `ceil`, `trunc`
- Complex: `conjugate`, `realPart`, `imagPart`, `csqrt`, `clog`, `angle`
- All trig/log functions support complex arrays

### Reductions
- `sum`, `prod`, `mean`, `variance`, `std` (with `axis` and `keepdims`)
- `min`, `max`, `argmin`, `argmax` (with `axis` and `keepdims`)
- `all`, `any`, `ptp`, `median`, `percentile`, `quantile`
- `cumsum`, `cumprod`, `diff` (with `axis` parameter)
- `allclose` for tolerance-based comparison

### Linear Algebra
- `dot`, `matmul`, `inner`, `outer`, `cross` (with complex support)

### Broadcasting
- Full broadcasting support for both real and complex arrays

## Known Differences from NumPy

1. **No view semantics**: All operations create copies. Reshape, transpose, and slicing allocate new arrays.
2. **No boolean dtype**: Comparison operations return float arrays with 0.0/1.0.
3. **Limited dtypes**: Only float64 and complex128 are supported.
4. **Global reductions**: `sum()`, `mean()`, `prod()` return `Double` for real arrays. Use `complexSum()` etc. for complex scalar results.
5. **No fancy indexing**: No boolean mask indexing or integer array indexing.
6. **No negative step slicing**: Only forward ranges are supported.

## Not Implemented

- `inv`, `solve`, `svd`, `eig`, `qr`, `cholesky`, `det` (linear algebra decompositions)
- FFT and signal processing
- `sort`, `argsort`, `searchsorted`, `unique`
- `nan*` reduction variants (`nanmean`, `nansum`, etc.)
- `.npy`/`.npz` file I/O
- Structured/record dtypes
- Advanced RNG with seeding and distributions
