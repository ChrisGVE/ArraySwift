# Changelog

All notable changes to ArraySwift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-29

### Added

#### Advanced Indexing (`NDArrayIndexing.swift`)
- `normalizeIndex(_:size:)`: public helper to convert negative indices (NumPy wrapping)
- `subscript(_:)`, `subscript(_:_:)`, `subscript(_:_:_:)`, `subscript([Int])`: all
  integer element subscripts now accept negative indices (wraps from end of axis)
- `subscript(mask:)` → `NDArray`: boolean gather — selects elements where a `.bool`
  NDArray is non-zero, returns a 1-D result preserving the source dtype
- `maskSet(_:value:)` / `booleanSet(_:to:)`: boolean scatter setter for `Double` scalars
- `booleanSetInt64(_:to:)`: boolean scatter for `.int64` arrays
- `booleanIndex(_:)`: named-function form of the boolean-gather subscript
- `subscript(indices:[Int])`: fancy (gather) indexing — collects elements at given flat
  indices (negative indices supported, repeats allowed)
- `subscript(ndIndices:NDArray)`: fancy indexing via a `.int64` NDArray index list

#### Set Operations (`NDArraySetOps.swift`)
- `NDArray.intersect1d(_:_:)`: sorted unique elements common to both arrays
- `NDArray.union1d(_:_:)`: sorted unique elements from either array
- `NDArray.setdiff1d(_:_:)`: sorted unique elements in `a` absent from `b`
- `NDArray.setxor1d(_:_:)`: sorted unique elements in exactly one of the arrays
- `NDArray.in1d(_:_:)`: membership test, returns `.bool` NDArray
- All set operations support `.float64` and `.int64` dtypes

#### FFT Family (`NDArrayFFT.swift`)
- `NDArray.fft(_:)`: 1-D complex DFT (forward, unnormalised), complex128 in/out
- `NDArray.ifft(_:)`: 1-D inverse DFT (normalised by 1/n)
- `NDArray.rfft(_:)`: 1-D real→complex DFT, returns `n/2+1` non-redundant bins
- `NDArray.fft2(_:)` / `NDArray.ifft2(_:)`: 2-D DFT via sequential row+column passes
- `NDArray.fftn(_:)` / `NDArray.ifftn(_:)`: N-D DFT via repeated 1-D passes
- `NDArray.fftfreq(_:d:)`: sample-frequency array for interpreting DFT output bins
- Uses `vDSP_DFT_zop_CreateSetupD` for SIMD-accelerated O(N log N) at power-of-2
  lengths; exact O(N²) direct DFT fallback for arbitrary lengths

#### Multi-Dtype Support
- `ArrayDType` cases: `.int64` (signed 64-bit), `.bool` (UInt8 0/1), `.date` (TimeInterval)
- `ArrayStorage` enum: type-safe backing union (`float64`, `int64`, `bool`, `complex128`, `date`)
- Typed initializers: `NDArray(shape:int64Data:)`, `NDArray(shape:boolData:)`, `NDArray(shape:dates:)`, `NDArray(shape:dateIntervals:)`
- Typed accessors: `.int64Data`, `.boolData`, `.dateIntervals` (return `nil` for wrong dtype)
- Factory methods: `NDArray.boolArray(shape:values:)`, `NDArray.int64Array(shape:values:)`
- `NDArrayError.dtypeMismatch(operation:lhs:rhs:)` for invalid dtype combinations

#### NumPy Promotion Rules
- `ArrayDType.promote(_:_:)`: bool < int64 < float64 < complex128 promotion table
- `ArrayDType.promoteOrNil(_:_:)`: returns `nil` when date is involved
- `ArrayDType.isInteger` property for bool and int64 dtypes
- Binary arithmetic ops (`+`, `-`, `*`, `/`) automatically promote operands to common dtype

#### Int64 Dtype
- Exact integer arithmetic with wrapping overflow (`&+`, `&-`, `&*`) semantics
- `NDArray.argmaxArray()` → scalar-shape int64 NDArray (array-returning argmax)
- `NDArray.argminArray()` → scalar-shape int64 NDArray (array-returning argmin)
- `NDArray.argsort()` → 1-D int64 NDArray of sort indices (stable sort)

#### Bool Dtype
- `logicalAnd`/`logicalOr`/`logicalXor`/`logicalNot` return `.bool` NDArray when both operands are `.bool`
- `isnan()`, `isinf()`, `isfinite()`, `isposinf()`, `isneginf()` now return `.bool` NDArrays
- `zeros(_:dtype:.bool)` and `ones(_:dtype:.bool)` factory support

#### Date Dtype
- `NDArray.addInterval(_:)` throws — adds `TimeInterval` seconds to each date element
- `NDArray.subtractDates(_:)` throws — subtracts date arrays → float64 duration array
- `NDArray.addArray(_:)` throws — enforces date cannot mix with other dtypes in arithmetic
- Comparison ops (`equal`, `less`, etc.) work on date arrays via `real` accessor

### Changed
- `NDArray.dtype` is now a computed property derived from `ArrayStorage` (no stored `dtype` field)
- `NDArray.real` and `NDArray.imag` are now computed-property shims over `ArrayStorage`
  — existing float64 and complex128 code is fully backward compatible
- `zeros(_:dtype:)` and `ones(_:dtype:)` extended to handle all five dtypes
- `full(_:value:dtype:)` extended to handle all five dtypes
- `broadcast()` and `broadcastTo()` visibility changed from `private` to `internal`
- `reshape`, `flatten`, `squeeze` preserve original `ArrayStorage` (no silent float64 promotion)

### Breaking Changes
- `NDArray.dtype` can no longer be assigned directly — it derives from storage
- `NDArray.real` setter for int64/bool arrays promotes storage to float64 (consistent with NumPy `astype` behavior)
- Arrays constructed via `init(shape:dtype:real:imag:)` with `.int64` or `.bool` dtype still
  store as float64 (use `init(shape:int64Data:)` / `init(shape:boolData:)` for typed storage)

### Added
- `NDArrayError` enum with cases for common failure modes (invalid shape, axis, broadcast, dtype)
- Centralized `normalizeAxis` helper supporting negative axis indexing across all APIs
- `.T` computed property as shorthand for `transpose()`
- Reshape with `-1` dimension inference (one unknown dimension inferred from total size)
- `keepdims` parameter for axis reductions: `sum`, `prod`, `mean`, `variance`, `std`, `min`, `max`
- `cumsum(axis:)`, `cumprod(axis:)`, `diff(n:axis:)` with shape-preserving axis parameter
- `ptp(axis:)` for peak-to-peak range along an axis
- `allclose` static method and free function for tolerance-based array comparison
- `meshgrid` `indexing` parameter supporting `"xy"` (Cartesian) and `"ij"` (matrix) conventions
- `realPart()`, `imagPart()`, `conjugate()` utility methods
- `isNonZero(at:)` helper for complex truthiness checks
- Complex broadcasting for all arithmetic operations (add, subtract, multiply, divide)
- Complex support for `dot`, `matmul`, `inner`, `outer`
- Complex support for `log2`, `log10`, `log1p`, `expm1`, `arcsin`, `arccos`, `arctan`
- Complex support for `diag` (both creation and extraction)
- DocC guide articles: NumPy Compatibility, Platform Support, Thread Safety, API Stability
- CI workflow for documentation generation
- CONTRIBUTING.md with development guidelines
- 41 new tests (148 total, up from 107) including NumPy cross-validation tests

### Changed
- Shape/real/imag properties now have `internal(set)` access (read-only from outside the module)
- Initializers validate shape product matches data count
- Nested array initializers detect ragged arrays
- Complex equality (`equal`/`notEqual`) now checks both real and imaginary parts
- Logical operations treat complex elements as non-zero when either part is non-zero
- `where` operation preserves complex dtype in output
- `broadcastTo` now propagates imaginary parts for complex arrays
- Axis reductions optimized to use direct index arithmetic instead of per-slice allocations
- Variance/std use two-pass algorithm with `ddof` validation
- Comprehensive DocC comments added to all public APIs

### Fixed
- Complex `diag` no longer silently drops imaginary parts
- `fatalError` in broadcast replaced with `precondition` for better diagnostics

## [0.1.0] - 2026-01-17

### Added

#### Core
- `NDArray` type with shape, dtype, and split real/imaginary storage
- `ArrayDType` enum supporting `float64` and `complex128` data types
- Row-major (C-style) memory layout with computed strides
- `Equatable`, `CustomStringConvertible`, and `CustomDebugStringConvertible` conformances

#### Array Creation
- `zeros`, `ones`, `full`, `empty` - filled arrays
- `eye`, `identity`, `diag` - identity and diagonal matrices
- `arange`, `linspace`, `logspace`, `geomspace` - sequences
- `random`, `randn`, `randint` - random arrays
- `zerosLike`, `onesLike`, `emptyLike`, `fullLike` - shape-matching arrays
- `complexArray`, `fromPolar`, `fromInterleaved` - complex array creation
- `meshgrid` - coordinate matrices
- Convenience initializers from Swift arrays (1D, 2D, 3D)

#### Array Manipulation
- `reshape`, `flatten`, `ravel` - shape changes
- `transpose`, `T`, `swapaxes`, `moveaxis` - axis reordering
- `squeeze`, `expandDims` - dimension adjustment
- `concatenate`, `stack`, `split` - combining and splitting
- `tile`, `repeat` - tiling
- `flip`, `roll` - element reordering
- `copy` - deep copy

#### Mathematical Functions
- Trigonometric: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`
- Exponential/logarithmic: `exp`, `log`, `log2`, `log10`, `log1p`, `expm1`
- Power: `sqrt`, `square`, `power`
- Rounding: `floor`, `ceil`, `round`, `trunc`
- Other: `abs`, `sign`, `clip`
- Complex: `angle`, `conjugate`, `csqrt`, `clog`

#### Arithmetic Operations
- Element-wise: `+`, `-`, `*`, `/` operators
- Scalar operations with Double
- Compound assignment: `+=`, `-=`, `*=`, `/=`
- Unary negation
- Broadcasting support for binary operations
- Full complex number arithmetic

#### Reduction Operations
- Basic: `sum`, `prod`, `mean`
- Statistics: `variance`, `std`, `median`, `percentile`, `quantile`
- Min/max: `min`, `max`, `argmin`, `argmax`, `ptp`
- Cumulative: `cumsum`, `cumprod`, `diff`
- Boolean: `all`, `any`
- Axis-based reductions for all operations
- Complex number support for reductions

#### Linear Algebra
- `dot` - generalized dot product (1D, 2D, mixed dimensions)
- `matmul` - matrix multiplication with vDSP optimization
- `cross` - 3D vector cross product
- `inner`, `outer` - inner and outer products

#### Comparison and Logic
- Element-wise: `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`
- Scalar comparisons
- Special values: `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`
- Logical: `logicalAnd`, `logicalOr`, `logicalXor`, `logicalNot`
- `where` - conditional element selection

#### Subscript Access
- Single element: `arr[i]`, `arr[row, col]`, `arr[d0, d1, d2]`
- N-dimensional: `arr[[i, j, k]]`
- Row/column slices: `arr[row: i]`, `arr[col: j]`
- Range slices: `arr[0..<5]`, `arr[0..<2, 1..<3]`
- Complex element access: `arr[complex: i]`

### Performance
- vDSP acceleration for vectorized operations
- Split complex storage for optimal vDSP_zD* function compatibility
- Efficient memory layout with contiguous storage

### Dependencies
- Swift 5.9+
- iOS 15+ / macOS 12+
- Apple Accelerate framework
