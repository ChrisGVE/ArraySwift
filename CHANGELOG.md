# Changelog

All notable changes to ArraySwift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
