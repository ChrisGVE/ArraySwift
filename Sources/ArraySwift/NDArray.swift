//
//  NDArray.swift
//  ArraySwift
//
//  Created by Christian C. Berclaz on 2026-01-05.
//  Copyright © 2026 Christian C. Berclaz. All rights reserved.
//
//  Licensed under the MIT License.
//

import Accelerate
import Foundation

// MARK: - Array Data Types

/// Data type enumeration for array elements
/// Supports real (float64) and complex (complex128) numbers
public enum ArrayDType: String, CaseIterable, Sendable {
  /// Real numbers - 64-bit floating point (default)
  case float64 = "float64"
  /// Complex numbers - two 64-bit floats (real + imaginary)
  case complex128 = "complex128"

  /// Bytes per element for this dtype
  public var bytesPerElement: Int {
    switch self {
    case .float64: return 8
    case .complex128: return 16
    }
  }

  /// Whether this dtype represents complex numbers
  public var isComplex: Bool { self == .complex128 }

  /// Create dtype from string, defaulting to float64 for unknown values
  public init(from string: String?) {
    switch string?.lowercased() {
    case "complex128", "complex": self = .complex128
    default: self = .float64
    }
  }
}

// MARK: - Error Types

/// Errors that can occur during NDArray operations
public enum NDArrayError: Error, CustomStringConvertible {
  /// Shape product does not match data element count
  case invalidShape(shape: [Int], dataCount: Int)
  /// Axis value is out of valid range for the array's dimensions
  case invalidAxis(axis: Int, ndim: Int)
  /// Shapes are incompatible for the requested operation
  case shapeMismatch(operation: String, shapes: [[Int]])
  /// Shapes cannot be broadcast together
  case broadcastFailure(shapes: [[Int]])
  /// Data type is not supported for the requested operation
  case unsupportedDtype(operation: String, dtype: ArrayDType)
  /// Array size is insufficient for the operation (e.g., ddof >= size)
  case insufficientSize(operation: String, size: Int, requirement: String)

  public var description: String {
    switch self {
    case .invalidShape(let shape, let dataCount):
      return "Shape \(shape) requires \(shape.reduce(1, *)) elements but got \(dataCount)"
    case .invalidAxis(let axis, let ndim):
      return "Axis \(axis) is out of bounds for array with \(ndim) dimensions"
    case .shapeMismatch(let operation, let shapes):
      return "\(operation): incompatible shapes \(shapes)"
    case .broadcastFailure(let shapes):
      return "Cannot broadcast shapes \(shapes)"
    case .unsupportedDtype(let operation, let dtype):
      return "\(operation) does not support dtype \(dtype)"
    case .insufficientSize(let operation, let size, let requirement):
      return "\(operation): array size \(size) does not satisfy \(requirement)"
    }
  }
}

// MARK: - N-Dimensional Array Structure

/// N-dimensional array representation with split storage for complex support
/// Uses separate real/imag arrays for optimal vDSP vectorization
public struct NDArray: Sendable {
  /// The shape of the array as a list of dimension sizes.
  public internal(set) var shape: [Int]

  /// The element type of the array.
  public internal(set) var dtype: ArrayDType

  /// Real part of the array data (always present).
  public internal(set) var real: [Double]

  /// Imaginary part of the array data (present only for `.complex128` arrays).
  public internal(set) var imag: [Double]?

  /// Number of dimensions (axes) in the array. Equivalent to NumPy's `ndarray.ndim`.
  public var ndim: Int { shape.count }

  /// Total number of elements in the array. Equivalent to NumPy's `ndarray.size`.
  public var size: Int { real.count }

  /// Whether this array stores complex numbers.
  public var isComplex: Bool { dtype.isComplex }

  /// Backward-compatibility alias for `real`. Equivalent to accessing the real part directly.
  public internal(set) var data: [Double] {
    get { real }
    set { real = newValue }
  }

  /// Row-major (C-style) strides for each dimension.
  ///
  /// Each entry gives the number of elements to skip in the flat storage to advance
  /// one step along that dimension. Equivalent to NumPy's `ndarray.strides` (in
  /// elements rather than bytes).
  public var strides: [Int] {
    var result = [Int](repeating: 1, count: shape.count)
    for i in stride(from: shape.count - 2, through: 0, by: -1) {
      result[i] = result[i + 1] * shape[i + 1]
    }
    return result
  }

  /// Convert a multi-dimensional index to a flat (linear) storage index.
  /// - Parameter indices: One index per dimension; must have the same count as `ndim`.
  /// - Returns: The corresponding flat index into `real` (and `imag`).
  public func flatIndex(_ indices: [Int]) -> Int {
    let strides = self.strides
    var index = 0
    for i in 0..<indices.count {
      index += indices[i] * strides[i]
    }
    return index
  }

  // MARK: - Initializers

  /// Create a real (`float64`) array from a flat data buffer.
  ///
  /// This is the primary backward-compatible initializer. The product of all values
  /// in `shape` must equal `data.count`.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - data: Flat element storage in row-major order.
  public init(shape: [Int], data: [Double]) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == data.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(data.count)"
    )
    self.shape = shape
    self.dtype = .float64
    self.real = data
    self.imag = nil
  }

  /// Create an array with an explicit dtype and separate real/imaginary buffers.
  ///
  /// For `.float64` arrays pass `nil` for `imag`. For `.complex128` arrays `imag`
  /// must have the same count as `real`.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - dtype: Element type (`.float64` or `.complex128`).
  ///   - real: Real part storage in row-major order.
  ///   - imag: Imaginary part storage, or `nil` for real arrays.
  public init(shape: [Int], dtype: ArrayDType, real: [Double], imag: [Double]?) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == real.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(real.count)"
    )
    if let imagPart = imag {
      precondition(
        imagPart.count == real.count,
        "Imaginary array count \(imagPart.count) must match real array count \(real.count)"
      )
    }
    self.shape = shape
    self.dtype = dtype
    self.real = real
    self.imag = imag
  }

  // MARK: - Factory Methods

  /// Create a real (`.float64`) array from a flat data buffer.
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - data: Flat element storage in row-major order.
  /// - Returns: A new real `NDArray`.
  public static func realArray(shape: [Int], data: [Double]) -> NDArray {
    NDArray(shape: shape, dtype: .float64, real: data, imag: nil)
  }

  /// Create a complex (`.complex128`) array from separate real and imaginary buffers.
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - real: Real part storage in row-major order.
  ///   - imag: Imaginary part storage (must have the same count as `real`).
  /// - Returns: A new complex `NDArray`.
  public static func complexArray(shape: [Int], real: [Double], imag: [Double]) -> NDArray {
    precondition(real.count == imag.count, "real and imag arrays must have same size")
    return NDArray(shape: shape, dtype: .complex128, real: real, imag: imag)
  }

  /// Create a `.complex128` array from interleaved format `[r0, i0, r1, i1, ...]`.
  ///
  /// Uses `vDSP_ctozD` for SIMD-optimised deinterleaving. The interleaved buffer length
  /// divided by two must equal the product of the values in `shape`.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - interleaved: Alternating real/imaginary values in row-major order.
  /// - Returns: A new `.complex128` `NDArray`.
  public static func fromInterleaved(shape: [Int], interleaved: [Double]) -> NDArray {
    precondition(interleaved.count % 2 == 0, "interleaved array must have even count")
    let count = interleaved.count / 2
    var realPart = [Double](repeating: 0, count: count)
    var imagPart = [Double](repeating: 0, count: count)

    // Use vDSP for efficient deinterleaving
    interleaved.withUnsafeBufferPointer { src in
      realPart.withUnsafeMutableBufferPointer { realBuf in
        imagPart.withUnsafeMutableBufferPointer { imagBuf in
          var split = DSPDoubleSplitComplex(
            realp: realBuf.baseAddress!,
            imagp: imagBuf.baseAddress!
          )
          // Convert from interleaved to split complex format
          vDSP_ctozD(
            UnsafePointer<DSPDoubleComplex>(OpaquePointer(src.baseAddress!)),
            2,  // stride in source (interleaved pairs)
            &split,
            1,  // stride in destination
            vDSP_Length(count)
          )
        }
      }
    }

    return NDArray(shape: shape, dtype: .complex128, real: realPart, imag: imagPart)
  }

  /// Convert a complex array to interleaved format `[r0, i0, r1, i1, ...]`.
  ///
  /// Uses `vDSP_ztocD` for SIMD-optimised interleaving.
  ///
  /// - Returns: A flat buffer of alternating real/imaginary values, or `nil` for real arrays.
  public func toInterleaved() -> [Double]? {
    guard isComplex, let imagPart = imag else { return nil }

    var result = [Double](repeating: 0, count: size * 2)

    real.withUnsafeBufferPointer { realBuf in
      imagPart.withUnsafeBufferPointer { imagBuf in
        var split = DSPDoubleSplitComplex(
          realp: UnsafeMutablePointer(mutating: realBuf.baseAddress!),
          imagp: UnsafeMutablePointer(mutating: imagBuf.baseAddress!)
        )
        result.withUnsafeMutableBufferPointer { dst in
          // Convert from split to interleaved complex format
          vDSP_ztocD(
            &split,
            1,  // stride in source
            UnsafeMutablePointer<DSPDoubleComplex>(OpaquePointer(dst.baseAddress!)),
            2,  // stride in destination (interleaved pairs)
            vDSP_Length(size)
          )
        }
      }
    }

    return result
  }

  // MARK: - Element Access

  /// Check if element at flat index is non-zero.
  /// For complex arrays, returns true if either real or imaginary part is non-zero.
  public func isNonZero(at index: Int) -> Bool {
    if real[index] != 0 { return true }
    if isComplex, let imagPart = imag, imagPart[index] != 0 { return true }
    return false
  }

  /// Get the real part of the element at the given flat index.
  /// - Parameter index: Flat (linear) index into the storage buffer.
  /// - Returns: The real component at that position.
  public func getReal(at index: Int) -> Double {
    real[index]
  }

  /// Get the complex element at the given flat index as a `(re, im)` tuple.
  ///
  /// For real arrays the imaginary component is always `0`.
  ///
  /// - Parameter index: Flat (linear) index into the storage buffer.
  /// - Returns: A tuple `(re:, im:)` with the real and imaginary components.
  public func getComplex(at index: Int) -> (re: Double, im: Double) {
    (real[index], imag?[index] ?? 0)
  }

  /// Set the real part of the element at the given flat index.
  /// - Parameters:
  ///   - index: Flat (linear) index into the storage buffer.
  ///   - value: New real value.
  public mutating func setReal(at index: Int, value: Double) {
    real[index] = value
  }

  /// Set a complex element at the given flat index.
  ///
  /// For real arrays the imaginary component is silently ignored.
  ///
  /// - Parameters:
  ///   - index: Flat (linear) index into the storage buffer.
  ///   - re: Real component.
  ///   - im: Imaginary component.
  public mutating func setComplex(at index: Int, re: Double, im: Double) {
    real[index] = re
    if imag != nil {
      imag![index] = im
    }
  }

  /// Return a copy of this array promoted to `.complex128` with a zero imaginary part.
  ///
  /// If the array is already complex, `self` is returned unchanged.
  ///
  /// - Returns: A `.complex128` array with the same real data and zeros for the imaginary part.
  public func promoteToComplex() -> NDArray {
    if isComplex { return self }
    return NDArray(
      shape: shape,
      dtype: .complex128,
      real: real,
      imag: [Double](repeating: 0, count: size)
    )
  }

  // MARK: - Subscript Operators

  /// Access a real element by flat (linear) index.
  ///
  /// Equivalent to `arr.real[index]`. For 1-D arrays this matches standard integer indexing.
  ///
  /// - Parameter index: Zero-based flat index.
  public subscript(index: Int) -> Double {
    get { real[index] }
    set { real[index] = newValue }
  }

  /// Access a real element in a 2-D array by row and column indices.
  /// - Parameters:
  ///   - row: Zero-based row index.
  ///   - col: Zero-based column index.
  public subscript(row: Int, col: Int) -> Double {
    get {
      precondition(ndim == 2, "2D subscript requires 2D array")
      return real[row * shape[1] + col]
    }
    set {
      precondition(ndim == 2, "2D subscript requires 2D array")
      real[row * shape[1] + col] = newValue
    }
  }

  /// Access a real element in a 3-D array.
  /// - Parameters:
  ///   - d0: Index along the first dimension.
  ///   - d1: Index along the second dimension.
  ///   - d2: Index along the third dimension.
  public subscript(d0: Int, d1: Int, d2: Int) -> Double {
    get {
      precondition(ndim == 3, "3D subscript requires 3D array")
      return real[d0 * strides[0] + d1 * strides[1] + d2]
    }
    set {
      precondition(ndim == 3, "3D subscript requires 3D array")
      real[d0 * strides[0] + d1 * strides[1] + d2] = newValue
    }
  }

  /// Access a real element using an array of N-dimensional indices.
  /// - Parameter indices: One index per dimension; count must equal `ndim`.
  public subscript(indices: [Int]) -> Double {
    get {
      precondition(indices.count == ndim, "Index count must match dimensions")
      return real[flatIndex(indices)]
    }
    set {
      precondition(indices.count == ndim, "Index count must match dimensions")
      real[flatIndex(indices)] = newValue
    }
  }

  /// Extract a single row from a 2-D array as a 1-D array.
  ///
  /// - Parameter row: Zero-based row index. Supports complex arrays.
  /// - Returns: A 1-D `NDArray` containing the row elements.
  public subscript(row row: Int) -> NDArray {
    get {
      precondition(ndim == 2, "Row subscript requires 2D array")
      let cols = shape[1]
      let start = row * cols
      let rowData = Array(real[start..<start + cols])
      if isComplex, let imagPart = imag {
        let rowImag = Array(imagPart[start..<start + cols])
        return NDArray(shape: [cols], dtype: .complex128, real: rowData, imag: rowImag)
      }
      return NDArray(shape: [cols], data: rowData)
    }
  }

  /// Extract a single column from a 2-D array as a 1-D array.
  ///
  /// - Parameter col: Zero-based column index. Supports complex arrays.
  /// - Returns: A 1-D `NDArray` containing the column elements.
  public subscript(col col: Int) -> NDArray {
    get {
      precondition(ndim == 2, "Column subscript requires 2D array")
      let rows = shape[0]
      let cols = shape[1]
      var colData = [Double](repeating: 0, count: rows)
      for i in 0..<rows {
        colData[i] = real[i * cols + col]
      }
      if isComplex, let imagPart = imag {
        var colImag = [Double](repeating: 0, count: rows)
        for i in 0..<rows {
          colImag[i] = imagPart[i * cols + col]
        }
        return NDArray(shape: [rows], dtype: .complex128, real: colData, imag: colImag)
      }
      return NDArray(shape: [rows], data: colData)
    }
  }

  /// Extract a contiguous slice of elements as a 1-D array.
  ///
  /// - Parameter range: Half-open range of flat indices. Supports complex arrays.
  /// - Returns: A 1-D `NDArray` with the specified elements.
  public subscript(range: Range<Int>) -> NDArray {
    get {
      let sliceData = Array(real[range])
      if isComplex, let imagPart = imag {
        let sliceImag = Array(imagPart[range])
        return NDArray(
          shape: [sliceData.count], dtype: .complex128, real: sliceData, imag: sliceImag)
      }
      return NDArray(shape: [sliceData.count], data: sliceData)
    }
  }

  /// Extract a sub-matrix from a 2-D array using row and column ranges.
  ///
  /// - Parameters:
  ///   - rowRange: Half-open range of row indices.
  ///   - colRange: Half-open range of column indices.
  /// - Returns: A 2-D `NDArray` sub-matrix. Supports complex arrays.
  public subscript(rowRange: Range<Int>, colRange: Range<Int>) -> NDArray {
    get {
      precondition(ndim == 2, "Range subscript requires 2D array")
      let cols = shape[1]
      let newRows = rowRange.count
      let newCols = colRange.count
      var resultData = [Double](repeating: 0, count: newRows * newCols)
      var resultImag: [Double]? = isComplex ? [Double](repeating: 0, count: newRows * newCols) : nil

      var idx = 0
      for r in rowRange {
        for c in colRange {
          resultData[idx] = real[r * cols + c]
          if isComplex, let imagPart = imag {
            resultImag![idx] = imagPart[r * cols + c]
          }
          idx += 1
        }
      }

      if isComplex {
        return NDArray(
          shape: [newRows, newCols], dtype: .complex128, real: resultData, imag: resultImag)
      }
      return NDArray(shape: [newRows, newCols], data: resultData)
    }
  }

  /// Access a complex element by flat index, returning a `(real:, imag:)` tuple.
  ///
  /// For real arrays the imaginary component is always `0`.
  ///
  /// - Parameter index: Flat (linear) index.
  public subscript(complex index: Int) -> (real: Double, imag: Double) {
    get {
      (real[index], imag?[index] ?? 0)
    }
    set {
      real[index] = newValue.real
      if isComplex {
        imag?[index] = newValue.imag
      }
    }
  }

  /// Access a complex element in a 2-D array by row/column, returning a `(real:, imag:)` tuple.
  ///
  /// - Parameters:
  ///   - row: Zero-based row index.
  ///   - col: Zero-based column index.
  public subscript(complex row: Int, _ col: Int) -> (real: Double, imag: Double) {
    get {
      precondition(ndim == 2, "2D subscript requires 2D array")
      let idx = row * shape[1] + col
      return (real[idx], imag?[idx] ?? 0)
    }
    set {
      precondition(ndim == 2, "2D subscript requires 2D array")
      let idx = row * shape[1] + col
      real[idx] = newValue.real
      if isComplex {
        imag?[idx] = newValue.imag
      }
    }
  }
}

// MARK: - Protocol Conformances

extension NDArray: Equatable {
  /// Returns `true` when both arrays have identical shape, dtype, and element values.
  ///
  /// For complex arrays both `real` and `imag` buffers are compared element-wise.
  public static func == (lhs: NDArray, rhs: NDArray) -> Bool {
    guard lhs.shape == rhs.shape else { return false }
    guard lhs.dtype == rhs.dtype else { return false }
    guard lhs.real == rhs.real else { return false }
    if lhs.isComplex || rhs.isComplex {
      guard lhs.imag == rhs.imag else { return false }
    }
    return true
  }
}

extension NDArray: CustomStringConvertible {
  /// A human-readable representation of the array, truncating large dimensions.
  ///
  /// Mirrors NumPy's default array print format: brackets around elements, `...` for
  /// large arrays, and a dtype suffix for complex arrays.
  public var description: String {
    if ndim == 0 {
      return "NDArray()"
    } else if ndim == 1 {
      return formatVector()
    } else if ndim == 2 {
      return formatMatrix()
    } else {
      return formatNDArray()
    }
  }

  private func formatVector() -> String {
    let maxElements = 10
    var parts: [String] = []

    if size <= maxElements {
      for i in 0..<size {
        parts.append(formatElement(i))
      }
    } else {
      for i in 0..<3 {
        parts.append(formatElement(i))
      }
      parts.append("...")
      for i in (size - 3)..<size {
        parts.append(formatElement(i))
      }
    }

    let dtype = isComplex ? ", dtype: complex128" : ""
    return "NDArray([\(parts.joined(separator: ", "))]\(dtype))"
  }

  private func formatMatrix() -> String {
    let rows = shape[0]
    let cols = shape[1]
    let maxRows = 6
    let maxCols = 10

    var lines: [String] = ["NDArray(["]

    let rowsToShow =
      rows <= maxRows ? Array(0..<rows) : Array(0..<3) + [-1] + Array((rows - 3)..<rows)
    let colsToShow =
      cols <= maxCols ? Array(0..<cols) : Array(0..<3) + [-1] + Array((cols - 3)..<cols)

    for (idx, row) in rowsToShow.enumerated() {
      if row == -1 {
        lines.append("  ...")
        continue
      }

      var rowParts: [String] = []
      for col in colsToShow {
        if col == -1 {
          rowParts.append("...")
        } else {
          rowParts.append(formatElement(row * cols + col))
        }
      }

      let prefix = idx == 0 ? " [" : "  ["
      let suffix = idx == rowsToShow.count - 1 ? "]" : "],"
      lines.append("\(prefix)\(rowParts.joined(separator: ", "))\(suffix)")
    }

    let dtype = isComplex ? ", dtype: complex128" : ""
    lines.append("]\(dtype), shape: \(shape))")
    return lines.joined(separator: "\n")
  }

  private func formatNDArray() -> String {
    let dtype = isComplex ? ", dtype: complex128" : ""
    return "NDArray(shape: \(shape)\(dtype), size: \(size))"
  }

  private func formatElement(_ index: Int) -> String {
    if isComplex, let imagPart = imag {
      let r = real[index]
      let i = imagPart[index]
      if i >= 0 {
        return String(format: "%.4g+%.4gi", r, i)
      } else {
        return String(format: "%.4g%.4gi", r, i)
      }
    } else {
      return String(format: "%.4g", real[index])
    }
  }
}

extension NDArray: CustomDebugStringConvertible {
  /// A detailed description including shape, dtype, size, and strides.
  public var debugDescription: String {
    let dtype = isComplex ? ", dtype: complex128" : ", dtype: float64"
    return "NDArray(shape: \(shape)\(dtype), size: \(size), strides: \(strides))"
  }
}

// MARK: - Axis Validation

extension NDArray {
  /// Normalize and validate an axis value, supporting negative indexing.
  /// Returns the normalized (non-negative) axis index.
  /// - Parameters:
  ///   - axis: The axis to validate (may be negative for counting from end)
  ///   - ndim: Number of dimensions to validate against (defaults to self.ndim)
  /// - Returns: The normalized axis index in range [0, ndim)
  public func normalizeAxis(_ axis: Int, ndim: Int? = nil) -> Int {
    let dims = ndim ?? self.ndim
    let normalized = axis < 0 ? axis + dims : axis
    precondition(
      normalized >= 0 && normalized < dims,
      "Axis \(axis) is out of bounds for array with \(dims) dimensions"
    )
    return normalized
  }

  /// Static axis normalization for use in class methods.
  internal static func normalizeAxis(_ axis: Int, ndim: Int) -> Int {
    let normalized = axis < 0 ? axis + ndim : axis
    precondition(
      normalized >= 0 && normalized < ndim,
      "Axis \(axis) is out of bounds for array with \(ndim) dimensions"
    )
    return normalized
  }
}

// MARK: - Type Alias for backward compatibility

/// Backward compatibility alias - ArrayData was the internal name
public typealias ArrayData = NDArray
