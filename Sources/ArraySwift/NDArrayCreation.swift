//
//  NDArrayCreation.swift
//  ArraySwift
//
//  Factory functions for creating NDArrays
//

import Accelerate
import Foundation

// MARK: - Factory Functions

extension NDArray {

  // MARK: - Basic Creation

  /// Create an array filled with zeros. Equivalent to NumPy's `numpy.zeros`.
  /// - Parameters:
  ///   - shape: Shape of the output array.
  ///   - dtype: Element type (default `.float64`). Pass `.complex128` for a complex array.
  /// - Returns: A new zero-filled `NDArray`.
  public static func zeros(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
    let size = shape.reduce(1, *)
    let real = [Double](repeating: 0, count: size)
    let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
    return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
  }

  /// Create an array filled with ones. Equivalent to NumPy's `numpy.ones`.
  /// - Parameters:
  ///   - shape: Shape of the output array.
  ///   - dtype: Element type (default `.float64`). Pass `.complex128` for a complex array.
  /// - Returns: A new one-filled `NDArray`.
  public static func ones(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
    let size = shape.reduce(1, *)
    let real = [Double](repeating: 1, count: size)
    let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
    return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
  }

  /// Create an array filled with a constant real value. Equivalent to NumPy's `numpy.full`.
  /// - Parameters:
  ///   - shape: Shape of the output array.
  ///   - value: Fill value.
  ///   - dtype: Element type (default `.float64`).
  /// - Returns: A new constant-filled `NDArray`.
  public static func full(_ shape: [Int], value: Double, dtype: ArrayDType = .float64) -> NDArray {
    let size = shape.reduce(1, *)
    let real = [Double](repeating: value, count: size)
    let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
    return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
  }

  /// Create a `.complex128` array filled with a constant complex value.
  /// - Parameters:
  ///   - shape: Shape of the output array.
  ///   - real: Real part of the fill value.
  ///   - imag: Imaginary part of the fill value.
  /// - Returns: A new constant complex `NDArray`.
  public static func full(_ shape: [Int], real: Double, imag: Double) -> NDArray {
    let size = shape.reduce(1, *)
    let realData = [Double](repeating: real, count: size)
    let imagData = [Double](repeating: imag, count: size)
    return NDArray(shape: shape, dtype: .complex128, real: realData, imag: imagData)
  }

  /// Create an "uninitialized" array of the given shape. Equivalent to NumPy's `numpy.empty`.
  ///
  /// In practice the storage is zero-filled. Callers should not rely on any particular initial
  /// value, matching NumPy's contract that empty arrays contain arbitrary data.
  ///
  /// - Parameters:
  ///   - shape: Shape of the output array.
  ///   - dtype: Element type (default `.float64`).
  /// - Returns: A new `NDArray` with unspecified element values.
  public static func empty(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
    zeros(shape, dtype: dtype)
  }

  // MARK: - Range Creation

  /// Create a 1-D array of evenly spaced values within a half-open interval.
  ///
  /// Equivalent to NumPy's `numpy.arange`. Values are computed as
  /// `start + i * step` for `i = 0, 1, 2, …` while the result is strictly within
  /// `[start, stop)` (or `(stop, start]` when `step < 0`). Index-based generation
  /// avoids floating-point accumulation errors.
  ///
  /// - Parameters:
  ///   - start: Start of the interval, inclusive (default `0`).
  ///   - stop: End of the interval, exclusive.
  ///   - step: Spacing between consecutive values (default `1`). Must be non-zero.
  /// - Returns: A 1-D `NDArray` of real values.
  public static func arange(start: Double = 0, stop: Double, step: Double = 1) -> NDArray {
    guard step != 0 else { return NDArray(shape: [0], data: []) }

    // Calculate count to avoid floating-point accumulation errors
    let count: Int
    if step > 0 {
      count = Swift.max(0, Int(Darwin.ceil((stop - start) / step)))
    } else {
      count = Swift.max(0, Int(Darwin.ceil((start - stop) / (-step))))
    }

    guard count > 0 else { return NDArray(shape: [0], data: []) }

    // Use index-based calculation to avoid accumulation errors
    var values = [Double](repeating: 0, count: count)
    for i in 0..<count {
      let value = start + Double(i) * step
      // Only include values that are strictly within range
      if step > 0 && value >= stop { break }
      if step < 0 && value <= stop { break }
      values[i] = value
    }

    // Trim to actual count if we broke early
    let actualCount = values.prefix { val in
      step > 0 ? (val < stop) : (val > stop)
    }.count
    if actualCount < count {
      values = Array(values.prefix(actualCount))
    }

    return NDArray(shape: [values.count], data: values)
  }

  /// Create a 1-D array of `num` evenly spaced values over a closed (or half-open) interval.
  ///
  /// Equivalent to NumPy's `numpy.linspace`. When `endpoint` is `true`, both `start`
  /// and `stop` are included and the spacing is `(stop - start) / (num - 1)`.
  /// When `endpoint` is `false`, `stop` is excluded and the spacing is
  /// `(stop - start) / num`.
  ///
  /// - Parameters:
  ///   - start: Starting value of the interval.
  ///   - stop: End value of the interval.
  ///   - num: Number of samples (default `50`). Must be positive.
  ///   - endpoint: Whether to include `stop` as the last sample (default `true`).
  /// - Returns: A 1-D `NDArray` of `num` real values.
  public static func linspace(start: Double, stop: Double, num: Int = 50, endpoint: Bool = true)
    -> NDArray
  {
    guard num > 0 else { return NDArray(shape: [0], data: []) }
    guard num > 1 else { return NDArray(shape: [1], data: [start]) }

    var data = [Double](repeating: 0, count: num)
    let divisor = endpoint ? Double(num - 1) : Double(num)

    for i in 0..<num {
      data[i] = start + (stop - start) * Double(i) / divisor
    }

    return NDArray(shape: [num], data: data)
  }

  /// Create a 1-D array of `num` values evenly spaced on a logarithmic scale.
  ///
  /// Equivalent to NumPy's `numpy.logspace`. Values are `base^linspace(start, stop, num)`.
  ///
  /// - Parameters:
  ///   - start: Exponent of the first value (`base^start`).
  ///   - stop: Exponent of the last value (`base^stop`).
  ///   - num: Number of samples (default `50`).
  ///   - base: Base of the logarithm (default `10`).
  /// - Returns: A 1-D `NDArray` of `num` real values.
  public static func logspace(start: Double, stop: Double, num: Int = 50, base: Double = 10)
    -> NDArray
  {
    let linear = linspace(start: start, stop: stop, num: num)
    var data = linear.real
    for i in 0..<data.count {
      data[i] = pow(base, data[i])
    }
    return NDArray(shape: [num], data: data)
  }

  /// Create a 1-D array of `num` values with geometric spacing between `start` and `stop`.
  ///
  /// Equivalent to NumPy's `numpy.geomspace`. Both `start` and `stop` must be positive.
  /// The ratio between consecutive values is constant: `(stop / start)^(1 / (num - 1))`.
  ///
  /// - Parameters:
  ///   - start: First value (must be positive).
  ///   - stop: Last value (must be positive).
  ///   - num: Number of samples (default `50`).
  /// - Returns: A 1-D `NDArray` of `num` geometrically spaced real values,
  ///   or an empty array when either `start` or `stop` is non-positive.
  public static func geomspace(start: Double, stop: Double, num: Int = 50) -> NDArray {
    guard start > 0 && stop > 0 else {
      return NDArray(shape: [0], data: [])
    }
    return logspace(start: Darwin.log10(start), stop: Darwin.log10(stop), num: num)
  }

  // MARK: - Identity and Diagonal

  /// Create an `n × n` identity matrix. Equivalent to NumPy's `numpy.eye`.
  /// - Parameters:
  ///   - n: Number of rows (and columns).
  ///   - dtype: Element type (default `.float64`).
  /// - Returns: A 2-D `NDArray` with ones on the main diagonal and zeros elsewhere.
  public static func eye(_ n: Int, dtype: ArrayDType = .float64) -> NDArray {
    var data = [Double](repeating: 0, count: n * n)
    for i in 0..<n {
      data[i * n + i] = 1
    }
    let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: n * n) : nil
    return NDArray(shape: [n, n], dtype: dtype, real: data, imag: imag)
  }

  /// Create an `n × n` identity matrix. Alias for ``eye(_:dtype:)``.
  ///
  /// Equivalent to NumPy's `numpy.identity`.
  ///
  /// - Parameters:
  ///   - n: Number of rows (and columns).
  ///   - dtype: Element type (default `.float64`).
  /// - Returns: A 2-D `NDArray` identity matrix.
  public static func identity(_ n: Int, dtype: ArrayDType = .float64) -> NDArray {
    eye(n, dtype: dtype)
  }

  /// Create a diagonal matrix from a 1-D array, or extract a diagonal from a 2-D array.
  ///
  /// Equivalent to NumPy's `numpy.diag`. Preserves complex dtype.
  ///
  /// - When `values` is 1-D: returns a square matrix with `values` placed on diagonal `k`.
  /// - When `values` is 2-D: returns the 1-D array of elements on diagonal `k`.
  ///
  /// - Parameters:
  ///   - values: Source array (must be 1-D or 2-D).
  ///   - k: Diagonal offset — `0` for the main diagonal, positive for super-diagonals,
  ///     negative for sub-diagonals (default `0`).
  /// - Returns: A diagonal matrix or the extracted diagonal values.
  public static func diag(_ values: NDArray, k: Int = 0) -> NDArray {
    if values.ndim == 1 {
      // Create diagonal matrix from 1D array
      let n = values.size + Swift.abs(k)
      var data = [Double](repeating: 0, count: n * n)
      var imagData: [Double]? =
        values.isComplex
        ? [Double](repeating: 0, count: n * n) : nil

      for i in 0..<values.size {
        let row = k >= 0 ? i : i - k
        let col = k >= 0 ? i + k : i
        if row < n && col < n {
          data[row * n + col] = values.real[i]
          if values.isComplex, let imagPart = values.imag {
            imagData![row * n + col] = imagPart[i]
          }
        }
      }

      if values.isComplex {
        return NDArray(
          shape: [n, n], dtype: .complex128, real: data, imag: imagData)
      }
      return NDArray(shape: [n, n], data: data)
    } else if values.ndim == 2 {
      // Extract diagonal from 2D matrix
      let rows = values.shape[0]
      let cols = values.shape[1]

      var diagValues: [Double] = []
      var diagImag: [Double]? = values.isComplex ? [] : nil
      let startRow = k >= 0 ? 0 : -k
      let startCol = k >= 0 ? k : 0

      var row = startRow
      var col = startCol
      while row < rows && col < cols {
        diagValues.append(values.real[row * cols + col])
        if values.isComplex, let imagPart = values.imag {
          diagImag!.append(imagPart[row * cols + col])
        }
        row += 1
        col += 1
      }

      if values.isComplex {
        return NDArray(
          shape: [diagValues.count], dtype: .complex128, real: diagValues,
          imag: diagImag)
      }
      return NDArray(shape: [diagValues.count], data: diagValues)
    }

    return values
  }

  // MARK: - Random Arrays

  /// Create an array of random values drawn from a uniform distribution on `[0, 1)`.
  ///
  /// Equivalent to NumPy's `numpy.random.random`.
  ///
  /// - Parameter shape: Shape of the output array.
  /// - Returns: A real `NDArray` with values in `[0, 1)`.
  public static func random(_ shape: [Int]) -> NDArray {
    let size = shape.reduce(1, *)
    var data = [Double](repeating: 0, count: size)
    for i in 0..<size {
      data[i] = Double.random(in: 0..<1)
    }
    return NDArray(shape: shape, data: data)
  }

  /// Create an array of random values drawn from the standard normal distribution (μ=0, σ=1).
  ///
  /// Uses the Box-Muller transform. Equivalent to NumPy's `numpy.random.randn`.
  ///
  /// - Parameter shape: Shape of the output array.
  /// - Returns: A real `NDArray` with normally distributed values.
  public static func randn(_ shape: [Int]) -> NDArray {
    let size = shape.reduce(1, *)
    var data = [Double](repeating: 0, count: size)

    // Box-Muller transform for normal distribution
    for i in Swift.stride(from: 0, to: size - 1, by: 2) {
      let u1 = Double.random(in: Double.leastNonzeroMagnitude..<1)
      let u2 = Double.random(in: 0..<1)
      let r = Darwin.sqrt(-2 * Darwin.log(u1))
      let theta = 2 * .pi * u2
      data[i] = r * Darwin.cos(theta)
      data[i + 1] = r * Darwin.sin(theta)
    }

    // Handle odd size
    if size % 2 == 1 {
      let u1 = Double.random(in: Double.leastNonzeroMagnitude..<1)
      let u2 = Double.random(in: 0..<1)
      let r = Darwin.sqrt(-2 * Darwin.log(u1))
      let theta = 2 * .pi * u2
      data[size - 1] = r * Darwin.cos(theta)
    }

    return NDArray(shape: shape, data: data)
  }

  /// Create an array of random integers drawn uniformly from `[low, high)`.
  ///
  /// Equivalent to NumPy's `numpy.random.randint`. Values are stored as `Double`.
  ///
  /// - Parameters:
  ///   - low: Lower bound (inclusive).
  ///   - high: Upper bound (exclusive).
  ///   - shape: Shape of the output array.
  /// - Returns: A real `NDArray` with random integer values cast to `Double`.
  public static func randint(low: Int, high: Int, shape: [Int]) -> NDArray {
    let size = shape.reduce(1, *)
    var data = [Double](repeating: 0, count: size)
    for i in 0..<size {
      data[i] = Double(Int.random(in: low..<high))
    }
    return NDArray(shape: shape, data: data)
  }

  // MARK: - Like Functions

  /// Create a zero-filled array with the same shape and dtype as `other`.
  /// Equivalent to NumPy's `numpy.zeros_like`.
  /// - Parameter other: Array whose shape and dtype are used.
  /// - Returns: A new zero-filled `NDArray`.
  public static func zerosLike(_ other: NDArray) -> NDArray {
    zeros(other.shape, dtype: other.dtype)
  }

  /// Create a one-filled array with the same shape and dtype as `other`.
  /// Equivalent to NumPy's `numpy.ones_like`.
  /// - Parameter other: Array whose shape and dtype are used.
  /// - Returns: A new one-filled `NDArray`.
  public static func onesLike(_ other: NDArray) -> NDArray {
    ones(other.shape, dtype: other.dtype)
  }

  /// Create an "uninitialized" array with the same shape and dtype as `other`.
  /// Equivalent to NumPy's `numpy.empty_like`.
  /// - Parameter other: Array whose shape and dtype are used.
  /// - Returns: A new `NDArray` with unspecified element values.
  public static func emptyLike(_ other: NDArray) -> NDArray {
    empty(other.shape, dtype: other.dtype)
  }

  /// Create a constant-filled array with the same shape and dtype as `other`.
  /// Equivalent to NumPy's `numpy.full_like`.
  /// - Parameters:
  ///   - other: Array whose shape and dtype are used.
  ///   - value: Fill value.
  /// - Returns: A new constant-filled `NDArray`.
  public static func fullLike(_ other: NDArray, value: Double) -> NDArray {
    full(other.shape, value: value, dtype: other.dtype)
  }

  // MARK: - Complex Array Creation

  /// Create a `.complex128` array from two real arrays representing real and imaginary parts.
  ///
  /// Equivalent to NumPy's `real + 1j * imag`. Both arrays must have identical shapes.
  ///
  /// - Parameters:
  ///   - real: Array of real components.
  ///   - imag: Array of imaginary components.
  /// - Returns: A new `.complex128` `NDArray`.
  public static func complexArray(real: NDArray, imag: NDArray) -> NDArray {
    precondition(real.shape == imag.shape, "real and imag must have same shape")
    return NDArray(shape: real.shape, dtype: .complex128, real: real.real, imag: imag.real)
  }

  /// Create a `.complex128` array from polar coordinates (magnitude and phase angle).
  ///
  /// Each element is computed as `magnitude * (cos(phase) + i * sin(phase))`.
  /// Both arrays must have identical shapes.
  ///
  /// - Parameters:
  ///   - magnitude: Array of magnitudes (radii). Non-negative values expected.
  ///   - phase: Array of phase angles in radians.
  /// - Returns: A new `.complex128` `NDArray`.
  public static func fromPolar(magnitude: NDArray, phase: NDArray) -> NDArray {
    precondition(magnitude.shape == phase.shape, "magnitude and phase must have same shape")

    var realData = [Double](repeating: 0, count: magnitude.size)
    var imagData = [Double](repeating: 0, count: magnitude.size)

    for i in 0..<magnitude.size {
      let r = magnitude.real[i]
      let theta = phase.real[i]
      realData[i] = r * Darwin.cos(theta)
      imagData[i] = r * Darwin.sin(theta)
    }

    return NDArray(shape: magnitude.shape, dtype: .complex128, real: realData, imag: imagData)
  }

  // MARK: - Meshgrid

  /// Create coordinate matrices from coordinate vectors.
  ///
  /// Mirrors NumPy's `meshgrid` with support for both Cartesian (`"xy"`) and
  /// matrix (`"ij"`) index conventions.
  ///
  /// - Parameters:
  ///   - x: 1D array of x (or first) coordinates.
  ///   - y: 1D array of y (or second) coordinates.
  ///   - indexing: Index convention. `"xy"` (default) produces Cartesian output
  ///     where X varies along columns and Y along rows, matching standard plot
  ///     axes. `"ij"` produces matrix output where the first input varies along
  ///     rows and the second along columns.
  /// - Returns: Tuple of (X, Y) coordinate arrays with shape `[ny, nx]` for
  ///   `"xy"` indexing or `[nx, ny]` for `"ij"` indexing.
  public static func meshgrid(
    x: NDArray,
    y: NDArray,
    indexing: String = "xy"
  ) -> (NDArray, NDArray) {
    let nx = x.size
    let ny = y.size

    if indexing == "ij" {
      // Matrix indexing: first input (x) along rows, second (y) along columns.
      // Output shape: [nx, ny]
      var xData = [Double](repeating: 0, count: nx * ny)
      var yData = [Double](repeating: 0, count: nx * ny)

      for i in 0..<nx {
        for j in 0..<ny {
          xData[i * ny + j] = x.real[i]
          yData[i * ny + j] = y.real[j]
        }
      }

      return (NDArray(shape: [nx, ny], data: xData), NDArray(shape: [nx, ny], data: yData))
    }

    // Cartesian indexing (default "xy"): X varies along columns, Y along rows.
    // Output shape: [ny, nx]
    var xData = [Double](repeating: 0, count: ny * nx)
    var yData = [Double](repeating: 0, count: ny * nx)

    for j in 0..<ny {
      for i in 0..<nx {
        xData[j * nx + i] = x.real[i]
        yData[j * nx + i] = y.real[j]
      }
    }

    return (NDArray(shape: [ny, nx], data: xData), NDArray(shape: [ny, nx], data: yData))
  }
}

// MARK: - Convenience Initializers

extension NDArray {

  /// Create a 1-D `NDArray` directly from a Swift `[Double]`.
  /// - Parameter data: Elements in order.
  public init(_ data: [Double]) {
    self.init(shape: [data.count], data: data)
  }

  /// Create a 2-D `NDArray` from a row-major nested Swift array.
  ///
  /// All rows must have the same length; ragged arrays are rejected at runtime.
  ///
  /// - Parameter data: 2-D nested array where `data[row][col]` gives the element value.
  public init(_ data: [[Double]]) {
    let rows = data.count
    let cols = data.first?.count ?? 0
    for (i, row) in data.enumerated() {
      precondition(
        row.count == cols,
        "Ragged array: row \(i) has \(row.count) elements, expected \(cols)"
      )
    }
    var flat: [Double] = []
    flat.reserveCapacity(rows * cols)
    for row in data {
      flat.append(contentsOf: row)
    }
    self.init(shape: [rows, cols], data: flat)
  }

  /// Create a 3-D `NDArray` from a depth-major nested Swift array.
  ///
  /// All planes must have the same number of rows, and all rows must have the same
  /// number of columns; ragged arrays are rejected at runtime.
  ///
  /// - Parameter data: 3-D nested array where `data[plane][row][col]` gives the element value.
  public init(_ data: [[[Double]]]) {
    let d0 = data.count
    let d1 = data.first?.count ?? 0
    let d2 = data.first?.first?.count ?? 0
    for (i, plane) in data.enumerated() {
      precondition(
        plane.count == d1,
        "Ragged array: plane \(i) has \(plane.count) rows, expected \(d1)"
      )
      for (j, row) in plane.enumerated() {
        precondition(
          row.count == d2,
          "Ragged array: plane \(i) row \(j) has \(row.count) elements, expected \(d2)"
        )
      }
    }
    var flat: [Double] = []
    flat.reserveCapacity(d0 * d1 * d2)
    for plane in data {
      for row in plane {
        flat.append(contentsOf: row)
      }
    }
    self.init(shape: [d0, d1, d2], data: flat)
  }
}
