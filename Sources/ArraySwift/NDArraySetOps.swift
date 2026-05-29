//
//  NDArraySetOps.swift
//  ArraySwift
//
//  1-D set operations mirroring NumPy's numpy.lib.arraysetops:
//  intersect1d, union1d, setdiff1d, setxor1d, in1d.
//
//  Supports float64 and int64 dtypes. Results are sorted unique values.
//

import Foundation

// MARK: - Set operations

extension NDArray {

  // MARK: - intersect1d

  /// Sorted unique elements present in both `a` and `b`.
  ///
  /// Equivalent to NumPy's `numpy.intersect1d`. Works with `.float64` and `.int64` dtypes.
  /// The result dtype matches the inputs (both must have the same dtype).
  ///
  /// - Parameters:
  ///   - a: First 1-D input array.
  ///   - b: Second 1-D input array.
  /// - Returns: A sorted 1-D array of common unique elements.
  public static func intersect1d(_ a: NDArray, _ b: NDArray) -> NDArray {
    switch (a.storage, b.storage) {
    case (.int64(let da), .int64(let db)):
      let sa = Set(da)
      let sb = Set(db)
      let common = sa.intersection(sb).sorted()
      return NDArray(shape: [common.count], int64Data: common)
    default:
      let sa = Set(a.real)
      let sb = Set(b.real)
      let common = sa.intersection(sb).sorted()
      return NDArray(shape: [common.count], storage: .float64(common))
    }
  }

  // MARK: - union1d

  /// Sorted unique elements from either `a` or `b`.
  ///
  /// Equivalent to NumPy's `numpy.union1d`. Works with `.float64` and `.int64`.
  ///
  /// - Parameters:
  ///   - a: First 1-D input array.
  ///   - b: Second 1-D input array.
  /// - Returns: A sorted 1-D array of all unique elements from both inputs.
  public static func union1d(_ a: NDArray, _ b: NDArray) -> NDArray {
    switch (a.storage, b.storage) {
    case (.int64(let da), .int64(let db)):
      let combined = Set(da).union(Set(db)).sorted()
      return NDArray(shape: [combined.count], int64Data: combined)
    default:
      let combined = Set(a.real).union(Set(b.real)).sorted()
      return NDArray(shape: [combined.count], storage: .float64(combined))
    }
  }

  // MARK: - setdiff1d

  /// Sorted unique elements in `a` that are not in `b`.
  ///
  /// Equivalent to NumPy's `numpy.setdiff1d`. Works with `.float64` and `.int64`.
  ///
  /// - Parameters:
  ///   - a: Input array.
  ///   - b: Values to exclude.
  /// - Returns: A sorted 1-D array of unique elements in `a` absent from `b`.
  public static func setdiff1d(_ a: NDArray, _ b: NDArray) -> NDArray {
    switch (a.storage, b.storage) {
    case (.int64(let da), .int64(let db)):
      let sb = Set(db)
      let diff = Set(da).subtracting(sb).sorted()
      return NDArray(shape: [diff.count], int64Data: diff)
    default:
      let sb = Set(b.real)
      let diff = Set(a.real).subtracting(sb).sorted()
      return NDArray(shape: [diff.count], storage: .float64(diff))
    }
  }

  // MARK: - setxor1d

  /// Sorted unique elements in either `a` or `b` but not both.
  ///
  /// Equivalent to NumPy's `numpy.setxor1d`. Works with `.float64` and `.int64`.
  ///
  /// - Parameters:
  ///   - a: First 1-D input array.
  ///   - b: Second 1-D input array.
  /// - Returns: A sorted 1-D array of symmetrically different unique elements.
  public static func setxor1d(_ a: NDArray, _ b: NDArray) -> NDArray {
    switch (a.storage, b.storage) {
    case (.int64(let da), .int64(let db)):
      let sa = Set(da)
      let sb = Set(db)
      let xor = sa.symmetricDifference(sb).sorted()
      return NDArray(shape: [xor.count], int64Data: xor)
    default:
      let sa = Set(a.real)
      let sb = Set(b.real)
      let xor = sa.symmetricDifference(sb).sorted()
      return NDArray(shape: [xor.count], storage: .float64(xor))
    }
  }

  // MARK: - in1d

  /// Boolean membership test: for each element of `ar1`, is it in `ar2`?
  ///
  /// Equivalent to NumPy's `numpy.in1d`. Works with `.float64` and `.int64`.
  ///
  /// - Parameters:
  ///   - ar1: 1-D input array to test.
  ///   - ar2: The set of values to test membership against.
  /// - Returns: A `.bool` NDArray of the same size as `ar1` with 1 where the element
  ///   is found in `ar2` and 0 otherwise.
  public static func in1d(_ ar1: NDArray, _ ar2: NDArray) -> NDArray {
    switch (ar1.storage, ar2.storage) {
    case (.int64(let da), .int64(let db)):
      let lookup = Set(db)
      let flags: [UInt8] = da.map { lookup.contains($0) ? 1 : 0 }
      return NDArray(shape: [flags.count], storage: .bool(flags))
    default:
      let lookup = Set(ar2.real)
      let flags: [UInt8] = ar1.real.map { lookup.contains($0) ? 1 : 0 }
      return NDArray(shape: [flags.count], storage: .bool(flags))
    }
  }
}
