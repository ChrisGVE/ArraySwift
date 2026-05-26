# ArraySwift

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg?style=flat)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/ArraySwift?style=flat&logo=github)](https://github.com/ChrisGVE/ArraySwift/releases)
[![CI](https://github.com/ChrisGVE/ArraySwift/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisGVE/ArraySwift/actions/workflows/ci.yml)
[![SPM Compatible](https://img.shields.io/badge/SPM-Compatible-brightgreen.svg?style=flat&logo=swift&logoColor=white)](https://swift.org/package-manager/)
[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-F05138.svg?style=flat&logo=swift&logoColor=white)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2015+%20|%20macOS%2012+-007AFF.svg?style=flat&logo=apple&logoColor=white)](https://developer.apple.com)
[![Documentation](https://img.shields.io/badge/Documentation-DocC-blue.svg?style=flat&logo=readthedocs&logoColor=white)](https://swiftpackageindex.com/ChrisGVE/ArraySwift/documentation)

N-dimensional array library for Swift with NumPy-inspired API and Accelerate optimization.

## Overview

ArraySwift provides a powerful `NDArray` type for numerical computing in Swift. Designed with a familiar NumPy-style API, it leverages Apple's Accelerate framework for high-performance SIMD operations.

### Key Features

- **NumPy-inspired API** - Function names and behaviors mirror NumPy for a familiar experience
- **vDSP acceleration** - Leverages Apple's Accelerate framework for optimized vector operations
- **Complex number support** - First-class `complex128` dtype with split real/imaginary storage for optimal vDSP vectorization
- **Comprehensive operations** - Array creation, manipulation, math functions, arithmetic, reductions, and linear algebra
- **Pure Swift** - No external dependencies except Darwin/Accelerate

## Installation

Add ArraySwift to your Swift package:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/ArraySwift.git", from: "0.1.0")
]
```

Then add it as a dependency to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["ArraySwift"]
)
```

## Quick Start

```swift
import ArraySwift

// Create arrays
let a = NDArray.arange(start: 0, stop: 6).reshape([2, 3])
let b = NDArray.ones([2, 3])

// Element-wise operations
let c = a + b
let d = a * 2.0

// Reductions
let total = a.sum()           // Sum all elements
let rowMeans = a.mean(axis: 1) // Mean along rows

// Linear algebra
let matrix = NDArray([[1.0, 2.0], [3.0, 4.0]])
let vector = NDArray([1.0, 0.0])
let result = matrix.dot(vector)

// Complex numbers
let complex = NDArray.complexArray(
    shape: [2],
    real: [1.0, 2.0],
    imag: [3.0, 4.0]
)
let magnitude = complex.abs()
```

## Module Overview

| Module | Purpose |
|--------|--------|
| `NDArray` | Core N-dimensional array type with shape, dtype, and data storage |
| Creation | `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `logspace`, `geomspace`, `random`, `randn`, `meshgrid` |
| Manipulation | `reshape` (with -1 inference), `transpose`/`.T`, `flatten`, `concatenate`, `stack`, `split`, `flip`, `roll` |
| Math | `sin`, `cos`, `tan`, `exp`, `log`, `log2`, `log10`, `sqrt`, `abs`, `power`, `arcsin`, `arccos`, `arctan` (all with complex support) |
| Arithmetic | Element-wise `+`, `-`, `*`, `/`, broadcasting (real and complex), compound assignment |
| Reductions | `sum`, `mean`, `min`, `max`, `std`, `variance`, `prod`, `cumsum`, `cumprod`, `diff` (with `axis` and `keepdims`) |
| Linear Algebra | `dot`, `matmul`, `cross`, `inner`, `outer` (with complex support) |
| Comparison | `equal`, `less`, `greater`, `where`, logical operations, `allclose` (all complex-aware) |

## Data Types

ArraySwift supports two data types:

| Type | Description |
|------|-------------|
| `float64` | 64-bit floating point (default) |
| `complex128` | Complex numbers with 64-bit real and imaginary parts |

Complex arrays use split storage (separate real/imag arrays) for optimal vDSP vectorization performance.

## Examples

### Array Creation

```swift
// From Swift arrays
let arr1D = NDArray([1.0, 2.0, 3.0])
let arr2D = NDArray([[1.0, 2.0], [3.0, 4.0]])

// Factory functions
let zeros = NDArray.zeros([3, 4])
let ones = NDArray.ones([2, 2])
let eye = NDArray.eye(3)
let range = NDArray.arange(start: 0, stop: 10, step: 2)
let linear = NDArray.linspace(start: 0, stop: 1, num: 5)
```

### Shape Manipulation

```swift
let arr = NDArray.arange(start: 0, stop: 12)
let matrix = arr.reshape([3, -1])  // Infer columns: [3, 4]
let transposed = matrix.T          // Transpose shorthand
let flat = matrix.flatten()
```

### Reductions

```swift
let arr = NDArray([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

arr.sum()                            // 21.0
arr.sum(axis: 0)                     // [5, 7, 9]
arr.sum(axis: 1, keepdims: true)     // [[6], [15]] (shape [2,1])
arr.mean(axis: -1)                   // Negative axis: [2, 5]
arr.cumsum(axis: 1)                  // Cumulative sum along columns

// Tolerance comparison
let a = NDArray([1.0, 2.0, 3.0])
let b = NDArray([1.0, 2.0, 3.0 + 1e-9])
NDArray.allclose(a, b)               // true
```

### Linear Algebra

```swift
// Dot product
let a = NDArray([1.0, 2.0, 3.0])
let b = NDArray([4.0, 5.0, 6.0])
let dot = a.dot(b)  // 32.0

// Matrix multiplication
let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
let B = NDArray([[5.0, 6.0], [7.0, 8.0]])
let C = A.matmul(B)

// Cross product
let x = NDArray([1.0, 0.0, 0.0])
let y = NDArray([0.0, 1.0, 0.0])
let z = x.cross(y)  // [0, 0, 1]
```

### Complex Numbers

```swift
let complex = NDArray.complexArray(
    shape: [3],
    real: [1.0, 2.0, 3.0],
    imag: [4.0, 5.0, 6.0]
)

complex.dtype         // .complex128
complex.conjugate()   // Complex conjugate
complex.abs()         // Magnitude
complex.angle()       // Phase angle
complex.realPart()    // Extract real components
complex.imagPart()    // Extract imaginary components

// Complex broadcasting works
let scalar = NDArray.complexArray(shape: [1], real: [2], imag: [0])
let scaled = complex * scalar  // Element-wise with broadcasting

// Complex math functions
complex.sin()         // Complex sine
complex.log2()        // Complex log base 2
complex.arcsin()      // Complex inverse sine

// Complex comparisons check both parts
let a = NDArray.complexArray(shape: [2], real: [1, 2], imag: [3, 4])
let b = NDArray.complexArray(shape: [2], real: [1, 2], imag: [3, 0])
a.equal(b)            // [1.0, 0.0] — second element differs in imag
```

## Requirements

- Swift 5.9+
- iOS 15+ / macOS 12+
- Xcode 15+

## Related Libraries

ArraySwift is part of a suite of Swift scientific computing libraries:

- **NumericSwift** - Scientific computing (SciPy-inspired)
- **PlotSwift** - Data visualization

## License

ArraySwift is available under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
