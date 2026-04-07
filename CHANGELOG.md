# Changelog

All notable changes to ArraySwift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
