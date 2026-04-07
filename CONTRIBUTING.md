# Contributing to ArraySwift

Thank you for taking the time to contribute. This document covers everything you need to get a change from idea to merged PR.

---

## Quick start

```bash
# Clone the repository
git clone https://github.com/ChrisGVE/ArraySwift.git
cd ArraySwift

# Build
swift build

# Run the full test suite
swift test

# Run only the ArraySwift tests
swift test --filter ArraySwiftTests

# Build for release (optional, for performance profiling)
swift build -c release
```

Requirements: Swift 5.9+, Xcode 15+ or Swift toolchain on macOS 12+ / iOS 15+. No additional dependencies beyond the Apple Accelerate framework.

---

## Project layout

```
Sources/ArraySwift/
├── NDArray.swift             # Core struct, ArrayDType, storage layout
├── NDArrayCreation.swift     # zeros, ones, arange, linspace, eye, …
├── NDArrayManipulation.swift # reshape, transpose, concatenate, slicing, …
├── NDArrayMath.swift         # Element-wise math (sin, cos, sqrt, log, …)
├── NDArrayArithmetic.swift   # Operators (+, -, *, /) and broadcasting
└── NDArrayReduction.swift    # sum, mean, min, max, std, var, cumsum, …

Tests/ArraySwiftTests/
└── ArraySwiftTests.swift     # Single test file containing all test cases
```

Each source file maps to a single functional category. New operations belong in the file that matches their category; add a new file only when the category is genuinely new.

---

## Code style

### Swift conventions

- Follow the [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/) for naming, labels, and documentation.
- 4-space indentation; no trailing whitespace.
- Opening braces on the same line as the declaration.
- Use `// MARK: -` to group related functions within a file, matching the style already in place.
- Keep files under 400 lines and functions under 50 lines. If a function grows beyond that, break it into private helpers.

### Naming

ArraySwift mirrors NumPy's naming where there is a direct analogue:

| NumPy | ArraySwift |
|---|---|
| `numpy.zeros` | `NDArray.zeros` |
| `numpy.arange` | `NDArray.arange` |
| `numpy.sqrt` | `.sqrt()` |
| `numpy.sum(axis=0)` | `.sum(axis: 0)` |

When NumPy uses a snake_case name, translate it to camelCase (`log1p` stays `log1p`; `arctan2` stays `arctan2`). Do not invent names that differ from NumPy without a strong Swift-idiom reason.

### Complex number support

Every operation that is mathematically defined for complex numbers must handle `dtype == .complex128`. Silently dropping the imaginary part and returning a real result is a bug, not a default. When complex behaviour differs from real behaviour, document it in the DocC comment.

### Accelerate / vDSP

Prefer vDSP or vForce functions over manual loops whenever one exists. Use `DSPDoubleSplitComplex` for complex operations. Avoid materialising large intermediate arrays when a strided or in-place operation is possible.

---

## Documentation

Every public symbol requires a DocC comment. The minimum structure is:

```swift
/// One-sentence summary. Equivalent to NumPy's `numpy.function_name` where applicable.
///
/// Explain any non-obvious behaviour (NaN handling, complex input, axis semantics).
///
/// - Parameters:
///   - axis: The axis along which to reduce. Pass `nil` to reduce over all elements.
/// - Returns: A description of what is returned.
/// - Throws: `NDArrayError.invalidAxis` if `axis` is out of bounds.
public func example(axis: Int? = nil) -> NDArray { … }
```

Use `///` triple-slash comments, not `/* */` block comments. Build the docs locally to confirm they render:

```bash
swift package generate-documentation --target ArraySwift
```

---

## Testing

### Requirements

- Every public function must have at least one test.
- Tests must cover the happy path, edge cases (empty arrays, single elements, negative values), and complex-dtype variants where the function supports them.
- NumPy's own test suite is a good source of edge-case ideas.
- All tests must pass before a PR is opened: `swift test`.

### Adding tests

Tests live in `Tests/ArraySwiftTests/ArraySwiftTests.swift`. Group related tests with `// MARK: -` comments matching the source file they cover. Use `XCTAssertEqual` with a tolerance for floating-point comparisons:

```swift
XCTAssertEqual(result.real[0], expected, accuracy: 1e-10)
```

For complex results, assert both the real and imaginary parts separately.

---

## How to add a new array operation

Follow these steps in order:

1. **Identify the right file.** Match the operation to its functional category (creation, manipulation, math, arithmetic, or reduction). If none fits, discuss in an issue before creating a new file.

2. **Study the NumPy equivalent.** Read the NumPy source and test suite to understand edge cases, axis semantics, and expected behaviour for complex inputs.

3. **Write the function with a DocC comment.** Add it to the appropriate `extension NDArray` block inside the matching file. Keep the function under 50 lines; extract private helpers if needed.

4. **Handle both dtypes.** The function must either support `complex128` correctly or explicitly document that it is real-only and why.

5. **Use Accelerate where possible.** Check vDSP and vForce for a vectorised primitive before writing a scalar loop.

6. **Write tests first (TDD).** Add at least three test cases before marking the implementation complete: a basic case, a shape/axis variant, and a complex-dtype case if applicable.

7. **Verify the full suite still passes.** `swift test` — no new warnings, no failures.

8. **Open a focused PR.** One operation or one coherent set of closely related operations per PR.

### Minimal example — adding `NDArray.sign()`

```swift
// In NDArrayMath.swift, under // MARK: - Basic Math

/// Element-wise sign function. Equivalent to NumPy's `numpy.sign`.
///
/// Returns -1, 0, or 1 for each element. For complex arrays returns
/// `z / |z|` (unit phasor), or 0 where `|z| == 0`.
///
/// - Returns: An `NDArray` with the same shape and dtype as the receiver.
public func sign() -> NDArray {
    if isComplex, let imagPart = imag {
        let magnitudes = self.abs().real
        let realOut = zip(real, magnitudes).map { r, m in m == 0 ? 0.0 : r / m }
        let imagOut = zip(imagPart, magnitudes).map { i, m in m == 0 ? 0.0 : i / m }
        return NDArray(shape: shape, dtype: .complex128, real: realOut, imag: imagOut)
    }
    let result = real.map { $0 < 0 ? -1.0 : ($0 > 0 ? 1.0 : 0.0) }
    return NDArray(shape: shape, data: result)
}
```

```swift
// In ArraySwiftTests.swift

// MARK: - sign

func testSignPositive() {
    let a = NDArray(shape: [3], data: [2.0, 0.0, -3.0])
    let s = a.sign()
    XCTAssertEqual(s.real, [1.0, 0.0, -1.0])
}

func testSignComplex() {
    let a = NDArray(shape: [2], dtype: .complex128, real: [3.0, 0.0], imag: [4.0, 0.0])
    let s = a.sign()
    XCTAssertEqual(s.real[0], 0.6, accuracy: 1e-10)
    XCTAssertEqual(s.imag![0], 0.8, accuracy: 1e-10)
    XCTAssertEqual(s.real[1], 0.0, accuracy: 1e-10)
}
```

---

## Pull request guidelines

- **One PR, one purpose.** A single new operation, a bug fix, a documentation update, or a refactor — not all at once.
- **Title format:** use a conventional commit prefix: `feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`. Example: `feat: add sign() for real and complex arrays`.
- **Description:** explain what the change does and reference the NumPy equivalent or the issue it closes.
- **No warnings.** `swift build 2>&1 | grep -i warning` must be empty after your change.
- **All tests pass.** Include output of `swift test` if the change is non-trivial.
- **DocC compiles.** Run `swift package generate-documentation --target ArraySwift` and confirm there are no errors.

---

## Issue labels

| Label | Meaning |
|---|---|
| `bug` | Incorrect output or crash for documented behaviour |
| `enhancement` | New operation or opt-in feature |
| `complex-support` | A function that silently drops the imaginary part or lacks complex handling |
| `performance` | Correct but slower than it should be |
| `documentation` | Missing or incorrect DocC comments, README, or guides |
| `good first issue` | Contained, well-scoped, suitable for a first contribution |
| `breaking change` | Alters a public API in a backward-incompatible way |

When filing a bug, include: the ArraySwift version, Swift version, platform, a minimal reproducer, the actual output, and the expected output (cite NumPy behaviour where relevant).

---

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE) that covers this project.
