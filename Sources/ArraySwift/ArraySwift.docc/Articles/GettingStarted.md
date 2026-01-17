# Getting Started with ArraySwift

Learn how to create and manipulate N-dimensional arrays.

## Overview

ArraySwift provides a NumPy-inspired API for numerical computing in Swift. This guide covers the basics of creating arrays, performing operations, and working with complex numbers.

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

## Creating Arrays

### From Swift Arrays

```swift
import ArraySwift

// 1D array
let a = NDArray([1.0, 2.0, 3.0, 4.0])

// 2D array
let b = NDArray([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])

// 3D array
let c = NDArray([[[1.0, 2.0], [3.0, 4.0]],
                 [[5.0, 6.0], [7.0, 8.0]]])
```

### Factory Functions

```swift
// Zeros and ones
let zeros = NDArray.zeros([3, 4])       // 3x4 array of zeros
let ones = NDArray.ones([2, 2])         // 2x2 array of ones
let filled = NDArray.full([3], value: 5) // [5, 5, 5]

// Sequences
let range = NDArray.arange(start: 0, stop: 10, step: 2)  // [0, 2, 4, 6, 8]
let linear = NDArray.linspace(start: 0, stop: 1, num: 5) // [0, 0.25, 0.5, 0.75, 1]

// Identity matrix
let eye = NDArray.eye(3)  // 3x3 identity

// Random
let uniform = NDArray.random([2, 3])  // Uniform [0, 1)
let normal = NDArray.randn([2, 3])    // Standard normal
```

## Array Properties

```swift
let arr = NDArray([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

arr.shape   // [2, 3]
arr.ndim    // 2
arr.size    // 6
arr.strides // [3, 1]
arr.dtype   // .float64
```

## Element Access

```swift
let arr = NDArray([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

// Single element
let val = arr[0, 1]  // 2.0

// Row and column slices
let row = arr[row: 0]     // [1, 2, 3]
let col = arr[col: 1]     // [2, 5]

// Range slicing
let sub = arr[0..<2, 1..<3]  // [[2, 3], [5, 6]]
```

## Arithmetic Operations

```swift
let a = NDArray([1.0, 2.0, 3.0])
let b = NDArray([4.0, 5.0, 6.0])

// Element-wise operations
let sum = a + b      // [5, 7, 9]
let diff = a - b     // [-3, -3, -3]
let prod = a * b     // [4, 10, 18]
let quot = a / b     // [0.25, 0.4, 0.5]

// Scalar operations
let scaled = a * 2.0 // [2, 4, 6]
let shifted = a + 1  // [2, 3, 4]

// In-place operations
var c = NDArray([1.0, 2.0, 3.0])
c += b  // c is now [5, 7, 9]
```

## Shape Manipulation

```swift
let arr = NDArray.arange(start: 0, stop: 12)

// Reshape
let matrix = arr.reshape([3, 4])

// Transpose
let transposed = matrix.transpose()
let T = matrix.T  // Shorthand

// Flatten
let flat = matrix.flatten()
```

## Reductions

```swift
let arr = NDArray([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

// Global reductions
arr.sum()      // 21.0
arr.mean()     // 3.5
arr.min()      // 1.0
arr.max()      // 6.0

// Along axis
arr.sum(axis: 0)   // [5, 7, 9]  - sum columns
arr.sum(axis: 1)   // [6, 15]   - sum rows
arr.mean(axis: 0)  // [2.5, 3.5, 4.5]
```

## Mathematical Functions

```swift
let arr = NDArray([0.0, Double.pi/4, Double.pi/2])

// Trigonometric
let sines = arr.sin()
let cosines = arr.cos()

// Exponential and logarithmic
let exps = arr.exp()
let logs = NDArray([1.0, 10.0, 100.0]).log()

// Rounding
let vals = NDArray([1.2, 2.5, 3.8])
vals.floor()  // [1, 2, 3]
vals.ceil()   // [2, 3, 4]
vals.round()  // [1, 2, 4]
```

## Linear Algebra

```swift
// Dot product (1D)
let a = NDArray([1.0, 2.0, 3.0])
let b = NDArray([4.0, 5.0, 6.0])
let dotProduct = a.dot(b)  // 32.0

// Matrix multiplication
let A = NDArray([[1.0, 2.0], [3.0, 4.0]])
let B = NDArray([[5.0, 6.0], [7.0, 8.0]])
let C = A.matmul(B)

// Cross product (3D)
let x = NDArray([1.0, 0.0, 0.0])
let y = NDArray([0.0, 1.0, 0.0])
let z = x.cross(y)  // [0, 0, 1]
```

## Complex Numbers

ArraySwift supports complex numbers with split storage for optimal vDSP performance:

```swift
// Create complex array
let complex = NDArray.complexArray(
    shape: [3],
    real: [1.0, 2.0, 3.0],
    imag: [4.0, 5.0, 6.0]
)

// Check type
complex.dtype      // .complex128
complex.isComplex  // true

// Access components
let re = complex.real  // [1, 2, 3]
let im = complex.imag  // [4, 5, 6]

// Complex arithmetic works automatically
let doubled = complex * 2.0

// Complex-specific operations
let conjugate = complex.conjugate()
let magnitude = complex.abs()
let phase = complex.angle()
```
