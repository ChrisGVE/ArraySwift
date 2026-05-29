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

/// Data type enumeration for array elements.
///
/// Supports real (`float64`), integer (`int64`), boolean (`bool`),
/// complex (`complex128`), and temporal (`date`) element types.
public enum ArrayDType: String, CaseIterable, Sendable {
  /// Real numbers — 64-bit floating point (default)
  case float64 = "float64"
  /// Exact integers — signed 64-bit
  case int64 = "int64"
  /// Boolean values — stored as `UInt8` (0 = false, 1 = true)
  case bool = "bool"
  /// Complex numbers — two 64-bit floats (real + imaginary)
  case complex128 = "complex128"
  /// Temporal values — `TimeInterval` (seconds since reference date)
  case date = "date"

  /// Bytes per element for this dtype.
  public var bytesPerElement: Int {
    switch self {
    case .float64: return 8
    case .int64: return 8
    case .bool: return 1
    case .complex128: return 16
    case .date: return 8
    }
  }

  /// Whether this dtype represents complex numbers.
  public var isComplex: Bool { self == .complex128 }

  /// Whether this dtype represents integer values (int64 or bool).
  public var isInteger: Bool { self == .int64 || self == .bool }

  /// Create dtype from string, defaulting to float64 for unknown values.
  public init(from string: String?) {
    switch string?.lowercased() {
    case "complex128", "complex": self = .complex128
    case "int64", "int": self = .int64
    case "bool", "boolean": self = .bool
    case "date": self = .date
    default: self = .float64
    }
  }

  // MARK: - NumPy-style dtype promotion

  /// Promote two dtypes to their common type following NumPy promotion rules.
  ///
  /// Hierarchy: bool < int64 < float64 < complex128. The `date` dtype does not
  /// participate in general promotion; use ``promoteOrNil(_:_:)`` when date may
  /// be involved.
  ///
  /// - Parameters:
  ///   - lhs: First dtype.
  ///   - rhs: Second dtype.
  /// - Returns: The promoted dtype.
  public static func promote(_ lhs: ArrayDType, _ rhs: ArrayDType) -> ArrayDType {
    if lhs == rhs { return lhs }
    // Rank: bool=0, int64=1, float64=2, complex128=3
    let rank: (ArrayDType) -> Int = { dt in
      switch dt {
      case .bool: return 0
      case .int64: return 1
      case .float64, .date: return 2
      case .complex128: return 3
      }
    }
    return rank(lhs) >= rank(rhs) ? lhs : rhs
  }

  /// Promote two dtypes, returning `nil` when either is `.date` and the other
  /// is not compatible for arithmetic (date cannot mix with int64/complex128).
  ///
  /// - Returns: Promoted dtype, or `nil` when mixing date with an incompatible type.
  public static func promoteOrNil(_ lhs: ArrayDType, _ rhs: ArrayDType) -> ArrayDType? {
    if lhs == .date || rhs == .date {
      // date+date is handled explicitly (subtraction → float64); all other combos are errors
      return nil
    }
    return promote(lhs, rhs)
  }
}

// MARK: - Array Storage

/// Type-safe backing storage for an `NDArray`.
///
/// Each case holds the contiguous element buffer for its dtype. Using an enum
/// avoids the cost of optional arrays for every dtype while keeping the public
/// `real`/`imag` API intact via computed-property shims on `NDArray`.
public enum ArrayStorage: Sendable {
  /// `[Double]` backing for `.float64` arrays.
  case float64([Double])
  /// `[Int64]` backing for `.int64` arrays.
  case int64([Int64])
  /// `[UInt8]` backing for `.bool` arrays (0 = false, 1 = true).
  case bool([UInt8])
  /// Split complex backing for `.complex128` arrays.
  case complex128(real: [Double], imag: [Double])
  /// `[Double]` (TimeInterval) backing for `.date` arrays.
  case date([Double])

  /// Number of elements in this storage buffer.
  public var count: Int {
    switch self {
    case .float64(let d): return d.count
    case .int64(let d): return d.count
    case .bool(let d): return d.count
    case .complex128(let r, _): return r.count
    case .date(let d): return d.count
    }
  }

  /// The `ArrayDType` corresponding to this storage case.
  public var dtype: ArrayDType {
    switch self {
    case .float64: return .float64
    case .int64: return .int64
    case .bool: return .bool
    case .complex128: return .complex128
    case .date: return .date
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
  /// Two dtypes cannot be combined in the requested operation
  case dtypeMismatch(operation: String, lhs: ArrayDType, rhs: ArrayDType)

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
    case .dtypeMismatch(let operation, let lhs, let rhs):
      return "\(operation): cannot mix dtype \(lhs) with \(rhs)"
    }
  }
}

// MARK: - N-Dimensional Array Structure

/// N-dimensional array representation with typed element storage.
///
/// Stores data in an ``ArrayStorage`` enum that provides type-safe backing for
/// `float64`, `int64`, `bool`, `complex128`, and `date` dtypes. The `real` and
/// `imag` computed properties maintain full backward compatibility with the
/// original split-storage API.
public struct NDArray: Sendable {
  /// The shape of the array as a list of dimension sizes.
  public internal(set) var shape: [Int]

  /// The element type of the array.
  public var dtype: ArrayDType { storage.dtype }

  /// The type-safe backing storage.
  public internal(set) var storage: ArrayStorage

  /// Real part of the array data as `Double` values.
  ///
  /// For `float64` and `date` arrays this is the actual stored data. For `int64`
  /// and `bool` arrays the integer values are cast to `Double` on each access.
  /// For `complex128` arrays this returns the real component buffer.
  public var real: [Double] {
    get {
      switch storage {
      case .float64(let d): return d
      case .int64(let d): return d.map { Double($0) }
      case .bool(let d): return d.map { Double($0) }
      case .complex128(let r, _): return r
      case .date(let d): return d
      }
    }
    set {
      switch storage {
      case .float64: storage = .float64(newValue)
      case .complex128(_, let im): storage = .complex128(real: newValue, imag: im)
      case .date: storage = .date(newValue)
      default:
        // int64/bool: store as float64 when set via real accessor
        storage = .float64(newValue)
      }
    }
  }

  /// Imaginary part for `.complex128` arrays; `nil` for all other dtypes.
  public var imag: [Double]? {
    get {
      if case .complex128(_, let im) = storage { return im }
      return nil
    }
    set {
      if case .complex128(let r, _) = storage, let newImag = newValue {
        storage = .complex128(real: r, imag: newImag)
      }
    }
  }

  /// Number of dimensions (axes) in the array. Equivalent to NumPy's `ndarray.ndim`.
  public var ndim: Int { shape.count }

  /// Total number of elements in the array. Equivalent to NumPy's `ndarray.size`.
  public var size: Int { storage.count }

  /// Whether this array stores complex numbers.
  public var isComplex: Bool { dtype.isComplex }

  /// Backward-compatibility alias for `real`. Equivalent to accessing the real part directly.
  public var data: [Double] {
    get { real }
    set { real = newValue }
  }

  // MARK: - Typed data accessors

  /// The underlying `[Int64]` storage for `.int64` arrays; `nil` for other dtypes.
  public var int64Data: [Int64]? {
    if case .int64(let d) = storage { return d }
    return nil
  }

  /// The underlying `[UInt8]` storage for `.bool` arrays (0/1); `nil` for other dtypes.
  public var boolData: [UInt8]? {
    if case .bool(let d) = storage { return d }
    return nil
  }

  /// The underlying `[Double]` (TimeInterval) storage for `.date` arrays; `nil` for other dtypes.
  public var dateIntervals: [Double]? {
    if case .date(let d) = storage { return d }
    return nil
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
    self.storage = .float64(data)
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
    switch dtype {
    case .complex128:
      let im = imag ?? [Double](repeating: 0, count: real.count)
      self.storage = .complex128(real: real, imag: im)
    case .date:
      self.storage = .date(real)
    default:
      self.storage = .float64(real)
    }
  }

  /// Create an `int64` array from a flat buffer of `Int64` values.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - int64Data: Flat element storage in row-major order.
  public init(shape: [Int], int64Data: [Int64]) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == int64Data.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(int64Data.count)"
    )
    self.shape = shape
    self.storage = .int64(int64Data)
  }

  /// Create a `bool` array from a flat buffer of `Bool` values.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - boolData: Flat element storage in row-major order.
  public init(shape: [Int], boolData: [Bool]) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == boolData.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(boolData.count)"
    )
    self.shape = shape
    self.storage = .bool(boolData.map { $0 ? 1 : 0 })
  }

  /// Create a `date` array from a flat buffer of `Date` values.
  ///
  /// Dates are stored as `TimeInterval` (seconds since the reference date).
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - dates: Flat element storage in row-major order.
  public init(shape: [Int], dates: [Date]) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == dates.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(dates.count)"
    )
    self.shape = shape
    self.storage = .date(dates.map { $0.timeIntervalSinceReferenceDate })
  }

  /// Create a `date` array from `TimeInterval` values directly.
  ///
  /// - Parameters:
  ///   - shape: Array dimensions.
  ///   - dateIntervals: Seconds since the reference date, row-major order.
  public init(shape: [Int], dateIntervals: [Double]) {
    let expectedSize = shape.reduce(1, *)
    precondition(
      expectedSize == dateIntervals.count,
      "Shape \(shape) requires \(expectedSize) elements but got \(dateIntervals.count)"
    )
    self.shape = shape
    self.storage = .date(dateIntervals)
  }

  /// Internal initializer from a pre-validated `ArrayStorage` value.
  internal init(shape: [Int], storage: ArrayStorage) {
    self.shape = shape
    self.storage = storage
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
    return NDArray(shape: shape, storage: .complex128(real: real, imag: imag))
  }

  /// Create a `.bool` array from a flat buffer of `Bool` values.
  public static func boolArray(shape: [Int], values: [Bool]) -> NDArray {
    NDArray(shape: shape, boolData: values)
  }

  /// Create an `.int64` array from a flat buffer of `Int64` values.
  public static func int64Array(shape: [Int], values: [Int64]) -> NDArray {
    NDArray(shape: shape, int64Data: values)
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
  ///
  /// For complex arrays, returns `true` if either real or imaginary part is non-zero.
  /// For bool arrays, returns `true` if the value is `1`.
  /// For int64 arrays, returns `true` if the value is non-zero.
  public func isNonZero(at index: Int) -> Bool {
    switch storage {
    case .float64(let d): return d[index] != 0
    case .int64(let d): return d[index] != 0
    case .bool(let d): return d[index] != 0
    case .complex128(let r, let im): return r[index] != 0 || im[index] != 0
    case .date(let d): return d[index] != 0
    }
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
    self[index] = value
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
    switch storage {
    case .complex128(var r, var i):
      r[index] = re
      i[index] = im
      storage = .complex128(real: r, imag: i)
    default:
      self[index] = re
    }
  }

  /// Return a copy of this array promoted to `.complex128` with a zero imaginary part.
  ///
  /// If the array is already complex, `self` is returned unchanged.
  ///
  /// - Returns: A `.complex128` array with the same real data and zeros for the imaginary part.
  public func promoteToComplex() -> NDArray {
    if isComplex { return self }
    let realVals = real
    return NDArray(
      shape: shape,
      storage: .complex128(real: realVals, imag: [Double](repeating: 0, count: size))
    )
  }

  /// Return a copy of this array promoted to `.float64`.
  ///
  /// For `float64` and `date` arrays, returns `self` (shares storage). For `int64`
  /// and `bool` arrays the values are cast to `Double`. For `complex128` arrays the
  /// imaginary part is discarded — use this only when you know the data is real.
  internal func toFloat64() -> NDArray {
    switch storage {
    case .float64: return self
    case .date: return NDArray(shape: shape, storage: .float64(real))
    default:
      return NDArray(shape: shape, storage: .float64(real))
    }
  }

  /// Return a copy promoted to the given dtype.
  ///
  /// - Parameter target: Destination dtype. Must be reachable from `self.dtype`
  ///   via the NumPy promotion table.
  internal func promoted(to target: ArrayDType) -> NDArray {
    if dtype == target { return self }
    switch target {
    case .float64:
      return NDArray(shape: shape, storage: .float64(real))
    case .int64:
      switch storage {
      case .bool(let d): return NDArray(shape: shape, int64Data: d.map { Int64($0) })
      default: return NDArray(shape: shape, int64Data: real.map { Int64($0) })
      }
    case .complex128:
      let r = real
      return NDArray(shape: shape, storage: .complex128(real: r, imag: [Double](repeating: 0, count: size)))
    default:
      return NDArray(shape: shape, storage: .float64(real))
    }
  }

  // MARK: - Subscript Operators

  /// Access a real element by flat (linear) index.
  ///
  /// Returns the element value as `Double` regardless of underlying dtype.
  /// For 1-D arrays this matches standard integer indexing.
  ///
  /// - Parameter index: Zero-based flat index.
  public subscript(index: Int) -> Double {
    get { real[index] }
    set {
      switch storage {
      case .float64(var d):
        d[index] = newValue
        storage = .float64(d)
      case .date(var d):
        d[index] = newValue
        storage = .date(d)
      default:
        // For int64/bool/complex, set via real accessor (promotes to float64)
        var r = real
        r[index] = newValue
        storage = .float64(r)
      }
    }
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
      self[row * shape[1] + col] = newValue
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
      self[d0 * strides[0] + d1 * strides[1] + d2] = newValue
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
      self[flatIndex(indices)] = newValue
    }
  }

  /// Extract a single row from a 2-D array as a 1-D array.
  ///
  /// - Parameter row: Zero-based row index. Supports all dtypes.
  /// - Returns: A 1-D `NDArray` containing the row elements.
  public subscript(row row: Int) -> NDArray {
    get {
      precondition(ndim == 2, "Row subscript requires 2D array")
      let cols = shape[1]
      let start = row * cols
      switch storage {
      case .float64(let d):
        return NDArray(shape: [cols], storage: .float64(Array(d[start..<start + cols])))
      case .int64(let d):
        return NDArray(shape: [cols], storage: .int64(Array(d[start..<start + cols])))
      case .bool(let d):
        return NDArray(shape: [cols], storage: .bool(Array(d[start..<start + cols])))
      case .complex128(let r, let im):
        return NDArray(shape: [cols], storage: .complex128(
          real: Array(r[start..<start + cols]),
          imag: Array(im[start..<start + cols])))
      case .date(let d):
        return NDArray(shape: [cols], storage: .date(Array(d[start..<start + cols])))
      }
    }
  }

  /// Extract a single column from a 2-D array as a 1-D array.
  ///
  /// - Parameter col: Zero-based column index. Supports all dtypes.
  /// - Returns: A 1-D `NDArray` containing the column elements.
  public subscript(col col: Int) -> NDArray {
    get {
      precondition(ndim == 2, "Column subscript requires 2D array")
      let rows = shape[0]
      let cols = shape[1]
      switch storage {
      case .float64(let d):
        return NDArray(shape: [rows], storage: .float64((0..<rows).map { d[$0 * cols + col] }))
      case .int64(let d):
        return NDArray(shape: [rows], storage: .int64((0..<rows).map { d[$0 * cols + col] }))
      case .bool(let d):
        return NDArray(shape: [rows], storage: .bool((0..<rows).map { d[$0 * cols + col] }))
      case .complex128(let r, let im):
        return NDArray(shape: [rows], storage: .complex128(
          real: (0..<rows).map { r[$0 * cols + col] },
          imag: (0..<rows).map { im[$0 * cols + col] }))
      case .date(let d):
        return NDArray(shape: [rows], storage: .date((0..<rows).map { d[$0 * cols + col] }))
      }
    }
  }

  /// Extract a contiguous slice of elements as a 1-D array.
  ///
  /// - Parameter range: Half-open range of flat indices. Supports all dtypes.
  /// - Returns: A 1-D `NDArray` with the specified elements.
  public subscript(range: Range<Int>) -> NDArray {
    get {
      switch storage {
      case .float64(let d):
        return NDArray(shape: [range.count], storage: .float64(Array(d[range])))
      case .int64(let d):
        return NDArray(shape: [range.count], storage: .int64(Array(d[range])))
      case .bool(let d):
        return NDArray(shape: [range.count], storage: .bool(Array(d[range])))
      case .complex128(let r, let im):
        return NDArray(shape: [range.count], storage: .complex128(
          real: Array(r[range]), imag: Array(im[range])))
      case .date(let d):
        return NDArray(shape: [range.count], storage: .date(Array(d[range])))
      }
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
      let newSize = newRows * newCols
      var idx = 0
      switch storage {
      case .float64(let d):
        var result = [Double](repeating: 0, count: newSize)
        for r in rowRange { for c in colRange { result[idx] = d[r * cols + c]; idx += 1 } }
        return NDArray(shape: [newRows, newCols], storage: .float64(result))
      case .int64(let d):
        var result = [Int64](repeating: 0, count: newSize)
        for r in rowRange { for c in colRange { result[idx] = d[r * cols + c]; idx += 1 } }
        return NDArray(shape: [newRows, newCols], storage: .int64(result))
      case .bool(let d):
        var result = [UInt8](repeating: 0, count: newSize)
        for r in rowRange { for c in colRange { result[idx] = d[r * cols + c]; idx += 1 } }
        return NDArray(shape: [newRows, newCols], storage: .bool(result))
      case .complex128(let r, let im):
        var rr = [Double](repeating: 0, count: newSize)
        var ri = [Double](repeating: 0, count: newSize)
        for row in rowRange { for c in colRange { rr[idx] = r[row * cols + c]; ri[idx] = im[row * cols + c]; idx += 1 } }
        return NDArray(shape: [newRows, newCols], storage: .complex128(real: rr, imag: ri))
      case .date(let d):
        var result = [Double](repeating: 0, count: newSize)
        for r in rowRange { for c in colRange { result[idx] = d[r * cols + c]; idx += 1 } }
        return NDArray(shape: [newRows, newCols], storage: .date(result))
      }
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
      switch storage {
      case .complex128(var r, var im):
        r[index] = newValue.real
        im[index] = newValue.imag
        storage = .complex128(real: r, imag: im)
      default:
        self[index] = newValue.real
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
      self[complex: idx] = newValue
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
    switch (lhs.storage, rhs.storage) {
    case (.float64(let a), .float64(let b)): return a == b
    case (.int64(let a), .int64(let b)): return a == b
    case (.bool(let a), .bool(let b)): return a == b
    case (.complex128(let ar, let ai), .complex128(let br, let bi)): return ar == br && ai == bi
    case (.date(let a), .date(let b)): return a == b
    default: return false
    }
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
      for i in 0..<size { parts.append(formatElement(i)) }
    } else {
      for i in 0..<3 { parts.append(formatElement(i)) }
      parts.append("...")
      for i in (size - 3)..<size { parts.append(formatElement(i)) }
    }

    let dtypeSuffix = dtype == .float64 ? "" : ", dtype: \(dtype.rawValue)"
    return "NDArray([\(parts.joined(separator: ", "))]\(dtypeSuffix))"
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

    let dtypeSuffix = dtype == .float64 ? "" : ", dtype: \(dtype.rawValue)"
    lines.append("]\(dtypeSuffix), shape: \(shape))")
    return lines.joined(separator: "\n")
  }

  private func formatNDArray() -> String {
    let dtypeSuffix = dtype == .float64 ? "" : ", dtype: \(dtype.rawValue)"
    return "NDArray(shape: \(shape)\(dtypeSuffix), size: \(size))"
  }

  private func formatElement(_ index: Int) -> String {
    switch storage {
    case .int64(let d): return "\(d[index])"
    case .bool(let d): return d[index] != 0 ? "true" : "false"
    case .complex128(let r, let im):
      let i = im[index]
      return i >= 0
        ? String(format: "%.4g+%.4gi", r[index], i)
        : String(format: "%.4g%.4gi", r[index], i)
    default: return String(format: "%.4g", real[index])
    }
  }
}

extension NDArray: CustomDebugStringConvertible {
  /// A detailed description including shape, dtype, size, and strides.
  public var debugDescription: String {
    return "NDArray(shape: \(shape), dtype: \(dtype.rawValue), size: \(size), strides: \(strides))"
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
