# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build & Test Commands

```bash
# Build the package
swift build

# Run all tests
swift test

# Run a specific test file
swift test --filter ArraySwiftTests

# Build for release
swift build -c release
```

## Project Overview

ArraySwift is a Swift library providing N-dimensional array functionality inspired by NumPy. It features split storage for optimal vDSP vectorization and supports both real (float64) and complex (complex128) data types.

### License

MIT License

### Design Philosophy

1. **NumPy-inspired API** - Function names and behaviors mirror NumPy patterns
2. **Pure Swift implementation** - No external dependencies except Darwin/Accelerate
3. **vDSP optimized** - Leverages Apple's Accelerate framework for performance
4. **Complex number support** - First-class complex128 dtype with split real/imag storage

## Project Structure

```
Sources/ArraySwift/
├── NDArray.swift           # Core NDArray struct and data types
├── NDArrayCreation.swift   # Array creation functions (zeros, ones, arange, etc.)
├── NDArrayManipulation.swift # Reshaping, transposing, slicing
├── NDArrayMath.swift       # Mathematical operations
├── NDArrayArithmetic.swift # Element-wise arithmetic operations
└── NDArrayReduction.swift  # Reduction operations (sum, mean, etc.)

Tests/ArraySwiftTests/
└── ArraySwiftTests.swift   # Test suite
```

## Module API Reference

### NDArray Type (`NDArray.swift`)

```swift
// Data types
public enum ArrayDType: String, CaseIterable, Sendable {
    case float64      // Real numbers (default)
    case complex128   // Complex numbers
}

// Core struct
public struct NDArray: Sendable {
    public var shape: [Int]
    public var dtype: ArrayDType
    public var real: [Double]
    public var imag: [Double]?

    public var ndim: Int      // Number of dimensions
    public var size: Int      // Total element count
    public var strides: [Int] // Row-major strides
    public var isComplex: Bool

    // Backward compatibility
    public var data: [Double] { get set }  // Alias for real
}

// Initializers
let arr = NDArray(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
let complex = NDArray(shape: [2], dtype: .complex128, real: [1, 2], imag: [3, 4])
```

### Array Creation (`NDArrayCreation.swift`)

```swift
// Factory functions
NDArray.zeros([3, 4])           // 3x4 array of zeros
NDArray.ones([2, 2])            // 2x2 array of ones
NDArray.full([2, 3], value: 5)  // Fill with value
NDArray.eye(3)                  // 3x3 identity matrix

// Sequences
NDArray.arange(0, 10, step: 2)  // [0, 2, 4, 6, 8]
NDArray.linspace(0, 1, count: 5) // [0, 0.25, 0.5, 0.75, 1]
NDArray.logspace(0, 2, count: 3) // [1, 10, 100]
```

### Array Manipulation (`NDArrayManipulation.swift`)

```swift
let arr = NDArray(shape: [2, 3], data: [...])

// Reshaping
arr.reshape([3, 2])
arr.flatten()
arr.ravel()

// Transposing
arr.transpose()
arr.T

// Slicing and indexing
arr[0, 1]              // Single element
arr[0..<2, 1..<3]      // Subarray
```

### Mathematical Operations (`NDArrayMath.swift`)

```swift
// Element-wise functions
arr.sin()
arr.cos()
arr.exp()
arr.log()
arr.sqrt()
arr.abs()

// Rounding
arr.round()
arr.floor()
arr.ceil()
```

### Arithmetic Operations (`NDArrayArithmetic.swift`)

```swift
// Element-wise arithmetic
let c = a + b
let d = a - b
let e = a * b
let f = a / b

// Scalar operations
let g = a + 5.0
let h = a * 2.0

// In-place operations
a += b
a *= 2.0
```

### Reduction Operations (`NDArrayReduction.swift`)

```swift
arr.sum()              // Sum of all elements
arr.sum(axis: 0)       // Sum along axis
arr.mean()             // Mean of all elements
arr.mean(axis: 1)      // Mean along axis
arr.min()              // Minimum value
arr.max()              // Maximum value
arr.prod()             // Product of elements
arr.std()              // Standard deviation
arr.var()              // Variance
```

## Integration with Sister Libraries

ArraySwift is part of a suite of Swift scientific computing libraries:

- **NumericSwift** - Can optionally depend on ArraySwift for enhanced array operations
- **PlotSwift** (planned) - Data visualization

To enable ArraySwift in NumericSwift:
```bash
NUMERICSWIFT_INCLUDE_ARRAYSWIFT=1 swift build
```

## Development Guidelines

1. **NumPy fidelity** - Study NumPy implementations for API design
2. **Test coverage** - Every function needs comprehensive tests
3. **Documentation** - DocC comments on all public functions
4. **Performance** - Use vDSP vectorization where beneficial
5. **Complex support** - Ensure all operations work with complex128 dtype

## Release Preparation Tasks

### Task 1: Create Comprehensive README.md

Create a professional README with:
- Badges (Swift version, platforms, SPM compatible, license, GitHub release, documentation)
- Overview and design philosophy
- Installation instructions (SPM)
- Quick start examples
- Module overview table
- Requirements section
- License section

Badge format to use:
```markdown
[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-F05138.svg?style=flat&logo=swift&logoColor=white)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2015+%20|%20macOS%2012+-007AFF.svg?style=flat&logo=apple&logoColor=white)](https://developer.apple.com)
[![SPM Compatible](https://img.shields.io/badge/SPM-Compatible-brightgreen.svg?style=flat&logo=swift&logoColor=white)](https://swift.org/package-manager/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/ArraySwift?style=flat&logo=github)](https://github.com/ChrisGVE/ArraySwift/releases)
[![Documentation](https://img.shields.io/badge/Documentation-DocC-blue.svg?style=flat&logo=readthedocs&logoColor=white)](https://github.com/ChrisGVE/ArraySwift)
```

### Task 2: Create DocC Documentation Catalog

Create directory structure:
```
Sources/ArraySwift/ArraySwift.docc/
├── Documentation.md          # Landing page
├── Articles/
│   ├── Installation.md
│   └── QuickStart.md
└── Modules/
    ├── NDArray.md
    ├── Creation.md
    ├── Manipulation.md
    ├── Math.md
    ├── Arithmetic.md
    └── Reduction.md
```

Add swift-docc-plugin to Package.swift:
```swift
.package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
```

### Task 3: Ensure Source Files Have DocC Comments

Review and add DocC comments to all public APIs in:
- NDArray.swift
- NDArrayCreation.swift
- NDArrayManipulation.swift
- NDArrayMath.swift
- NDArrayArithmetic.swift
- NDArrayReduction.swift

### Task 4: Test Coverage Audit

- Run `swift test` and verify all tests pass
- Review test coverage for edge cases
- Add tests for any untested functionality

### Task 5: Create CHANGELOG.md

Create CHANGELOG.md following Keep a Changelog format:
```markdown
# Changelog

All notable changes to ArraySwift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - YYYY-MM-DD

### Added
- Core `NDArray` type with float64 and complex128 support
- Array creation functions (zeros, ones, arange, linspace, etc.)
- Array manipulation (reshape, transpose, slicing)
- Mathematical operations (sin, cos, exp, log, etc.)
- Arithmetic operations (element-wise +, -, *, /)
- Reduction operations (sum, mean, min, max, std, var)
- vDSP acceleration for performance

### Dependencies
- Requires Swift 5.9+, iOS 15+ / macOS 12+
- Uses Apple Accelerate framework
```

### Task 6: Final Review and Release

1. Verify no compiler warnings: `swift build 2>&1 | grep -i warning`
2. Verify DocC generates: `swift package generate-documentation --target ArraySwift`
3. Push all changes to GitHub
4. Create release tag: `git tag 0.1.0 && git push origin 0.1.0`
5. Create GitHub release with release notes

## Test Statistics

- **Total tests**: TBD (audit needed)
- **Known failures**: TBD
