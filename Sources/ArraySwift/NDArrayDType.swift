//
//  NDArrayDType.swift
//  ArraySwift
//
//  Multi-dtype support: int64/bool arithmetic, date operations,
//  dtype-aware binary promotion, and dtype-specific reductions.
//

import Foundation

// MARK: - Dtype-aware binary arithmetic helpers

extension NDArray {

  // MARK: - Int64 element-wise operations

  /// Add two int64-backed arrays element-wise.
  internal static func int64Add(_ a: NDArray, _ b: NDArray) -> NDArray {
    guard case .int64(let da) = a.storage, case .int64(let db) = b.storage else {
      preconditionFailure("int64Add requires both arrays to be int64")
    }
    precondition(da.count == db.count, "Size mismatch in int64Add")
    return NDArray(shape: a.shape, int64Data: zip(da, db).map { $0 &+ $1 })
  }

  /// Subtract two int64-backed arrays element-wise.
  internal static func int64Sub(_ a: NDArray, _ b: NDArray) -> NDArray {
    guard case .int64(let da) = a.storage, case .int64(let db) = b.storage else {
      preconditionFailure("int64Sub requires both arrays to be int64")
    }
    precondition(da.count == db.count, "Size mismatch in int64Sub")
    return NDArray(shape: a.shape, int64Data: zip(da, db).map { $0 &- $1 })
  }

  /// Multiply two int64-backed arrays element-wise.
  internal static func int64Mul(_ a: NDArray, _ b: NDArray) -> NDArray {
    guard case .int64(let da) = a.storage, case .int64(let db) = b.storage else {
      preconditionFailure("int64Mul requires both arrays to be int64")
    }
    precondition(da.count == db.count, "Size mismatch in int64Mul")
    return NDArray(shape: a.shape, int64Data: zip(da, db).map { $0 &* $1 })
  }

  /// Divide two int64-backed arrays element-wise (truncating integer division).
  internal static func int64Div(_ a: NDArray, _ b: NDArray) -> NDArray {
    guard case .int64(let da) = a.storage, case .int64(let db) = b.storage else {
      preconditionFailure("int64Div requires both arrays to be int64")
    }
    precondition(da.count == db.count, "Size mismatch in int64Div")
    return NDArray(shape: a.shape, int64Data: zip(da, db).map { $0 / $1 })
  }

  // MARK: - Bool element-wise operations

  /// Element-wise logical AND.
  ///
  /// Returns a `.bool` NDArray when both inputs are `.bool`; returns a `.float64`
  /// NDArray otherwise (1.0 = true, 0.0 = false). Correctly handles complex arrays
  /// by treating an element as truthy if either real or imaginary part is non-zero.
  public func logicalAnd(_ other: NDArray) -> NDArray {
    if case .bool(let da) = storage, case .bool(let db) = other.storage {
      precondition(da.count == db.count, "Size mismatch in logicalAnd")
      return NDArray(shape: shape, storage: .bool(
        zip(da, db).map { ($0 != 0 && $1 != 0) ? UInt8(1) : UInt8(0) }))
    }
    // Keep original storage so isNonZero handles complex/int64 correctly
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      result[i] = (a.isNonZero(at: i) && b.isNonZero(at: i)) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical OR.
  ///
  /// Returns a `.bool` NDArray when both inputs are `.bool`; returns a `.float64`
  /// NDArray otherwise. Correctly handles complex arrays.
  public func logicalOr(_ other: NDArray) -> NDArray {
    if case .bool(let da) = storage, case .bool(let db) = other.storage {
      precondition(da.count == db.count, "Size mismatch in logicalOr")
      return NDArray(shape: shape, storage: .bool(
        zip(da, db).map { ($0 != 0 || $1 != 0) ? UInt8(1) : UInt8(0) }))
    }
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      result[i] = (a.isNonZero(at: i) || b.isNonZero(at: i)) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical XOR.
  ///
  /// Returns a `.bool` NDArray when both inputs are `.bool`; returns a `.float64`
  /// NDArray otherwise. Correctly handles complex arrays.
  public func logicalXor(_ other: NDArray) -> NDArray {
    if case .bool(let da) = storage, case .bool(let db) = other.storage {
      precondition(da.count == db.count, "Size mismatch in logicalXor")
      return NDArray(shape: shape, storage: .bool(
        zip(da, db).map { ($0 != 0) != ($1 != 0) ? UInt8(1) : UInt8(0) }))
    }
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      result[i] = (a.isNonZero(at: i) != b.isNonZero(at: i)) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical NOT.
  ///
  /// Returns a `.bool` NDArray when `self` is `.bool`; returns `.float64` otherwise.
  public func logicalNot() -> NDArray {
    if case .bool(let da) = storage {
      return NDArray(shape: shape, storage: .bool(da.map { $0 == 0 ? UInt8(1) : UInt8(0) }))
    }
    var result = [Double](repeating: 0, count: size)
    for i in 0..<size { result[i] = isNonZero(at: i) ? 0.0 : 1.0 }
    return NDArray(shape: shape, data: result)
  }

  // MARK: - Bool special-value checks

  /// Element-wise check for NaN values.
  ///
  /// Returns a `.bool` NDArray with 1 where NaN, 0 otherwise.
  public func isnan() -> NDArray {
    let result: [UInt8] = real.map { $0.isNaN ? 1 : 0 }
    return NDArray(shape: shape, storage: .bool(result))
  }

  /// Element-wise check for infinite values.
  ///
  /// Returns a `.bool` NDArray with 1 where infinite, 0 otherwise.
  public func isinf() -> NDArray {
    let result: [UInt8] = real.map { $0.isInfinite ? 1 : 0 }
    return NDArray(shape: shape, storage: .bool(result))
  }

  /// Element-wise check for finite values.
  ///
  /// Returns a `.bool` NDArray with 1 where finite, 0 otherwise.
  public func isfinite() -> NDArray {
    let result: [UInt8] = real.map { $0.isFinite ? 1 : 0 }
    return NDArray(shape: shape, storage: .bool(result))
  }

  /// Element-wise check for positive infinity.
  ///
  /// Returns a `.bool` NDArray with 1 where +inf, 0 otherwise.
  public func isposinf() -> NDArray {
    let result: [UInt8] = real.map { ($0.isInfinite && $0 > 0) ? 1 : 0 }
    return NDArray(shape: shape, storage: .bool(result))
  }

  /// Element-wise check for negative infinity.
  ///
  /// Returns a `.bool` NDArray with 1 where -inf, 0 otherwise.
  public func isneginf() -> NDArray {
    let result: [UInt8] = real.map { ($0.isInfinite && $0 < 0) ? 1 : 0 }
    return NDArray(shape: shape, storage: .bool(result))
  }

  // MARK: - Date operations

  /// Add a fixed time interval (in seconds) to each date element.
  ///
  /// - Parameter seconds: The number of seconds to add.
  /// - Returns: A new `.date` NDArray with the shifted values.
  /// - Throws: `NDArrayError.unsupportedDtype` if `self` is not `.date`.
  public func addInterval(_ seconds: Double) throws -> NDArray {
    guard case .date(let d) = storage else {
      throw NDArrayError.unsupportedDtype(operation: "addInterval", dtype: dtype)
    }
    return NDArray(shape: shape, storage: .date(d.map { $0 + seconds }))
  }

  /// Subtract another date array element-wise, returning durations in seconds.
  ///
  /// - Parameter other: Another `.date` array with the same shape.
  /// - Returns: A `.float64` NDArray of differences (seconds).
  /// - Throws: `NDArrayError.dtypeMismatch` if `other` is not `.date`;
  ///   `NDArrayError.shapeMismatch` if shapes differ.
  public func subtractDates(_ other: NDArray) throws -> NDArray {
    guard case .date(let da) = storage else {
      throw NDArrayError.unsupportedDtype(operation: "subtractDates", dtype: dtype)
    }
    guard case .date(let db) = other.storage else {
      throw NDArrayError.dtypeMismatch(operation: "subtractDates", lhs: dtype, rhs: other.dtype)
    }
    guard shape == other.shape else {
      throw NDArrayError.shapeMismatch(operation: "subtractDates", shapes: [shape, other.shape])
    }
    let result = zip(da, db).map { $0 - $1 }
    return NDArray(shape: shape, storage: .float64(result))
  }

  /// Add another array element-wise, with dtype promotion.
  ///
  /// This throwing variant enforces the promotion rules including date restrictions.
  /// For non-throwing use, the `+` operator automatically promotes.
  ///
  /// - Parameter other: Array to add.
  /// - Returns: A new NDArray with the promoted dtype.
  /// - Throws: `NDArrayError.dtypeMismatch` if date is mixed with an incompatible dtype.
  public func addArray(_ other: NDArray) throws -> NDArray {
    if dtype == .date || other.dtype == .date {
      throw NDArrayError.dtypeMismatch(operation: "addArray", lhs: dtype, rhs: other.dtype)
    }
    return add(other)
  }
}

// MARK: - Reductions producing int64

extension NDArray {

  /// Return the flat index of the maximum element as a scalar-shaped int64 NDArray.
  ///
  /// This is the array-returning complement of ``argmax()``.
  /// Equivalent to `numpy.argmax` when applied without an axis.
  ///
  /// - Returns: A shape-[1] `.int64` NDArray containing the index.
  public func argmaxArray() -> NDArray {
    let idx = argmax()
    return NDArray(shape: [1], int64Data: [Int64(idx)])
  }

  /// Return the flat index of the minimum element as a scalar-shaped int64 NDArray.
  ///
  /// This is the array-returning complement of ``argmin()``.
  /// Equivalent to `numpy.argmin` when applied without an axis.
  ///
  /// - Returns: A shape-[1] `.int64` NDArray containing the index.
  public func argminArray() -> NDArray {
    let idx = argmin()
    return NDArray(shape: [1], int64Data: [Int64(idx)])
  }

  /// Return the indices that would sort this array (ascending) as an int64 NDArray.
  ///
  /// Equivalent to NumPy's `numpy.argsort` applied to a flattened array.
  /// Uses a stable sort so equal elements retain their original order.
  ///
  /// - Returns: A 1-D `.int64` NDArray of indices.
  public func argsort() -> NDArray {
    let realVals = real
    let indices = (0..<size).sorted { realVals[$0] < realVals[$1] }
    return NDArray(shape: [size], int64Data: indices.map { Int64($0) })
  }
}

