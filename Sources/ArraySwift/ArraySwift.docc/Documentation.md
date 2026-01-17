# ``ArraySwift``

N-dimensional array library for Swift with NumPy-inspired API and Accelerate optimization.

## Overview

ArraySwift provides a powerful ``NDArray`` type for numerical computing in Swift. It features:

- **NumPy-inspired API** - Familiar function names and behaviors for Python developers
- **vDSP acceleration** - Leverages Apple's Accelerate framework for SIMD operations
- **Complex number support** - First-class `complex128` dtype with split real/imaginary storage
- **Comprehensive operations** - Creation, manipulation, math, arithmetic, and reductions

### Quick Example

```swift
import ArraySwift

// Create arrays
let a = NDArray.arange(start: 0, stop: 6).reshape([2, 3])
let b = NDArray.ones([2, 3])

// Element-wise operations
let c = a + b
let d = a * 2.0

// Reductions
let total = a.sum()
let rowMeans = a.mean(axis: 1)

// Linear algebra
let matrix = NDArray([[1.0, 2.0], [3.0, 4.0]])
let vector = NDArray([1.0, 0.0])
let result = matrix.dot(vector)
```

## Topics

### Essentials

- ``NDArray``
- ``ArrayDType``
- <doc:GettingStarted>

### Array Creation

- ``NDArray/zeros(_:dtype:)``
- ``NDArray/ones(_:dtype:)``
- ``NDArray/full(_:value:dtype:)``
- ``NDArray/eye(_:dtype:)``
- ``NDArray/arange(start:stop:step:)``
- ``NDArray/linspace(start:stop:num:endpoint:)``
- ``NDArray/random(_:)``
- ``NDArray/randn(_:)``

### Array Manipulation

- ``NDArray/reshape(_:)``
- ``NDArray/flatten()``
- ``NDArray/transpose()``
- ``NDArray/concatenate(_:axis:)``
- ``NDArray/stack(_:axis:)``
- ``NDArray/split(indices:axis:)``

### Mathematical Functions

- ``NDArray/sin()``
- ``NDArray/cos()``
- ``NDArray/exp()``
- ``NDArray/log()``
- ``NDArray/sqrt()``
- ``NDArray/abs()``
- ``NDArray/power(_:)-(Double)``

### Reduction Operations

- ``NDArray/sum(axis:)``
- ``NDArray/mean(axis:)``
- ``NDArray/min(axis:)``
- ``NDArray/max(axis:)``
- ``NDArray/std(axis:ddof:)``
- ``NDArray/variance(axis:ddof:)``

### Linear Algebra

- ``NDArray/dot(_:)``
- ``NDArray/matmul(_:)``
- ``NDArray/cross(_:)``
- ``NDArray/inner(_:)``
- ``NDArray/outer(_:)``

### Comparison and Logic

- ``NDArray/equal(_:)-(NDArray)``
- ``NDArray/less(_:)-(NDArray)``
- ``NDArray/greater(_:)-(NDArray)``
- ``NDArray/logicalAnd(_:)``
- ``NDArray/logicalOr(_:)``
- ``NDArray/where(_:_:_:)``
