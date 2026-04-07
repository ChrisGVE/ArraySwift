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
- ``NDArrayError``
- <doc:GettingStarted>

### Guides

- <doc:NumPyCompatibility>
- <doc:PlatformSupport>
- <doc:ThreadSafety>
- <doc:APIStability>

### Array Creation

- ``NDArray/zeros(_:dtype:)``
- ``NDArray/ones(_:dtype:)``
- ``NDArray/full(_:value:dtype:)``
- ``NDArray/eye(_:dtype:)``
- ``NDArray/arange(start:stop:step:)``
- ``NDArray/linspace(start:stop:num:endpoint:)``
- ``NDArray/logspace(start:stop:num:base:)``
- ``NDArray/geomspace(start:stop:num:)``
- ``NDArray/random(_:)``
- ``NDArray/randn(_:)``
- ``NDArray/randint(low:high:shape:)``
- ``NDArray/diag(_:k:)``
- ``NDArray/meshgrid(x:y:indexing:)``

### Array Manipulation

- ``NDArray/reshape(_:)``
- ``NDArray/flatten()``
- ``NDArray/transpose()``
- ``NDArray/T``
- ``NDArray/squeeze()``
- ``NDArray/expandDims(axis:)``
- ``NDArray/swapaxes(_:_:)``
- ``NDArray/moveaxis(source:destination:)``
- ``NDArray/concatenate(_:axis:)``
- ``NDArray/stack(_:axis:)``
- ``NDArray/split(indices:axis:)``
- ``NDArray/tile(_:)``
- ``NDArray/flip(axis:)``
- ``NDArray/roll(shift:axis:)``

### Mathematical Functions

- ``NDArray/sin()``
- ``NDArray/cos()``
- ``NDArray/tan()``
- ``NDArray/arcsin()``
- ``NDArray/arccos()``
- ``NDArray/arctan()``
- ``NDArray/exp()``
- ``NDArray/log()``
- ``NDArray/log2()``
- ``NDArray/log10()``
- ``NDArray/log1p()``
- ``NDArray/expm1()``
- ``NDArray/sqrt()``
- ``NDArray/abs()``
- ``NDArray/sign()``
- ``NDArray/conjugate()``
- ``NDArray/realPart()``
- ``NDArray/imagPart()``

### Reduction Operations

- ``NDArray/sum(axis:)``
- ``NDArray/sum(axis:keepdims:)``
- ``NDArray/mean(axis:)``
- ``NDArray/mean(axis:keepdims:)``
- ``NDArray/min(axis:)``
- ``NDArray/max(axis:)``
- ``NDArray/std(axis:ddof:)``
- ``NDArray/variance(axis:ddof:)``
- ``NDArray/prod(axis:)``
- ``NDArray/cumsum(axis:)``
- ``NDArray/cumprod(axis:)``
- ``NDArray/diff(n:axis:)``
- ``NDArray/ptp(axis:)``
- ``NDArray/argmin()``
- ``NDArray/argmax()``
- ``NDArray/median()``
- ``NDArray/allclose(_:_:rtol:atol:)``

### Linear Algebra

- ``NDArray/dot(_:)``
- ``NDArray/matmul(_:)``
- ``NDArray/cross(_:)``
- ``NDArray/inner(_:)``
- ``NDArray/outer(_:)``

### Comparison and Logic

- ``NDArray/equal(_:)-(NDArray)``
- ``NDArray/notEqual(_:)-(NDArray)``
- ``NDArray/less(_:)-(NDArray)``
- ``NDArray/lessEqual(_:)-(NDArray)``
- ``NDArray/greater(_:)-(NDArray)``
- ``NDArray/greaterEqual(_:)-(NDArray)``
- ``NDArray/logicalAnd(_:)``
- ``NDArray/logicalOr(_:)``
- ``NDArray/logicalXor(_:)``
- ``NDArray/logicalNot()``
- ``NDArray/where(_:_:_:)``
