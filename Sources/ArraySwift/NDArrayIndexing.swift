//
//  NDArrayIndexing.swift
//  ArraySwift
//
//  Negative-index normalisation, boolean masking, and fancy (gather) indexing.
//

import Foundation

// MARK: - Index Normalisation

extension NDArray {

  /// Normalise a flat index into the range `[0, size)`, supporting negative indices.
  ///
  /// A negative index counts backwards from the end of the buffer:
  /// `-1` → last element, `-size` → first element.
  ///
  /// - Parameters:
  ///   - index: The index to normalise (may be negative).
  ///   - size: The buffer length to normalise against.
  /// - Returns: A non-negative index in `[0, size)`.
  @inline(__always)
  public func normalizeIndex(_ index: Int, size: Int) -> Int {
    let n = index < 0 ? index + size : index
    precondition(n >= 0 && n < size, "Index \(index) is out of bounds for size \(size)")
    return n
  }
}

// MARK: - Negative-index element subscripts

extension NDArray {

  /// Access a real element by flat (linear) index, supporting negative indices.
  ///
  /// Negative indices wrap from the end: `-1` is the last element.
  ///
  /// - Parameter index: Zero-based or negative flat index.
  public subscript(index: Int) -> Double {
    get {
      let i = normalizeIndex(index, size: size)
      return real[i]
    }
    set {
      let i = normalizeIndex(index, size: size)
      switch storage {
      case .float64(var d):
        d[i] = newValue
        storage = .float64(d)
      case .date(var d):
        d[i] = newValue
        storage = .date(d)
      default:
        var r = real
        r[i] = newValue
        storage = .float64(r)
      }
    }
  }

  /// Access a real element in a 2-D array by row and column, supporting negative indices.
  ///
  /// - Parameters:
  ///   - row: Zero-based or negative row index.
  ///   - col: Zero-based or negative column index.
  public subscript(row: Int, col: Int) -> Double {
    get {
      precondition(ndim == 2, "2D subscript requires 2D array")
      let r = normalizeIndex(row, size: shape[0])
      let c = normalizeIndex(col, size: shape[1])
      return real[r * shape[1] + c]
    }
    set {
      precondition(ndim == 2, "2D subscript requires 2D array")
      let r = normalizeIndex(row, size: shape[0])
      let c = normalizeIndex(col, size: shape[1])
      self[r * shape[1] + c] = newValue
    }
  }

  /// Access a real element in a 3-D array, supporting negative indices.
  ///
  /// - Parameters:
  ///   - d0: Index along the first dimension (may be negative).
  ///   - d1: Index along the second dimension (may be negative).
  ///   - d2: Index along the third dimension (may be negative).
  public subscript(d0: Int, d1: Int, d2: Int) -> Double {
    get {
      precondition(ndim == 3, "3D subscript requires 3D array")
      let i0 = normalizeIndex(d0, size: shape[0])
      let i1 = normalizeIndex(d1, size: shape[1])
      let i2 = normalizeIndex(d2, size: shape[2])
      return real[i0 * strides[0] + i1 * strides[1] + i2]
    }
    set {
      precondition(ndim == 3, "3D subscript requires 3D array")
      let i0 = normalizeIndex(d0, size: shape[0])
      let i1 = normalizeIndex(d1, size: shape[1])
      let i2 = normalizeIndex(d2, size: shape[2])
      self[i0 * strides[0] + i1 * strides[1] + i2] = newValue
    }
  }

  /// Access a real element using an array of N-dimensional indices, supporting negatives.
  ///
  /// - Parameter indices: One index per dimension; count must equal `ndim`.
  public subscript(indices: [Int]) -> Double {
    get {
      precondition(indices.count == ndim, "Index count must match dimensions")
      let normalised = indices.enumerated().map { normalizeIndex($1, size: shape[$0]) }
      return real[flatIndex(normalised)]
    }
    set {
      precondition(indices.count == ndim, "Index count must match dimensions")
      let normalised = indices.enumerated().map { normalizeIndex($1, size: shape[$0]) }
      self[flatIndex(normalised)] = newValue
    }
  }
}

// MARK: - Boolean (mask) indexing

extension NDArray {

  /// Gather elements where a boolean mask is `true`, returning a 1-D array.
  ///
  /// Both `self` and `mask` must have the same `size`. The result dtype matches `self`.
  ///
  /// - Parameter mask: A `.bool` NDArray; non-zero values select elements.
  /// - Returns: A 1-D `NDArray` containing the selected elements.
  public func booleanIndex(_ mask: NDArray) -> NDArray {
    precondition(mask.dtype == .bool, "Boolean indexing requires a .bool mask")
    precondition(mask.size == size, "Mask size \(mask.size) must match array size \(size)")
    guard case .bool(let flags) = mask.storage else {
      preconditionFailure("Mask storage is not .bool")
    }
    let selected = flags.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
    return gatherElements(at: selected)
  }

  /// Set elements where a boolean mask is `true` to a `Double` scalar.
  ///
  /// - Parameters:
  ///   - mask: A `.bool` NDArray of the same size as `self`.
  ///   - value: Scalar to assign at each selected position.
  public mutating func booleanSet(_ mask: NDArray, to value: Double) {
    precondition(mask.dtype == .bool, "Boolean indexing requires a .bool mask")
    precondition(mask.size == size, "Mask size must match array size")
    guard case .bool(let flags) = mask.storage else {
      preconditionFailure("Mask storage is not .bool")
    }
    let selected = flags.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
    applyScalar(value, at: selected)
  }

  /// Set elements where a boolean mask is `true` to an `Int64` scalar.
  ///
  /// Requires `self` to be an `.int64` array.
  ///
  /// - Parameters:
  ///   - mask: A `.bool` NDArray of the same size as `self`.
  ///   - value: `Int64` scalar to assign at each selected position.
  public mutating func booleanSetInt64(_ mask: NDArray, to value: Int64) {
    precondition(dtype == .int64, "booleanSetInt64 requires an .int64 array")
    precondition(mask.dtype == .bool, "Mask must be .bool")
    precondition(mask.size == size, "Mask size must match array size")
    guard case .bool(let flags) = mask.storage, case .int64(var d) = storage else {
      preconditionFailure("Storage mismatch in booleanSetInt64")
    }
    for (i, flag) in flags.enumerated() where flag != 0 {
      d[i] = value
    }
    storage = .int64(d)
  }
}

// MARK: - Subscript convenience wrappers for boolean indexing

extension NDArray {

  /// Subscript form of ``booleanIndex(_:)``.
  ///
  /// Usage: `let selected = a[mask: boolArray]`
  ///
  /// - Parameter mask: A `.bool` NDArray of the same size.
  /// - Returns: 1-D `NDArray` of selected elements.
  public subscript(mask mask: NDArray) -> NDArray {
    booleanIndex(mask)
  }

  /// Subscript setter form of ``booleanSet(_:to:)`` for `Double` scalars.
  ///
  /// Usage: `a[mask: boolArray] = 99.0`
  ///
  /// Implemented as a separate named setter since Swift subscripts cannot have
  /// different return types for get/set on the same subscript label.
  public mutating func maskSet(_ mask: NDArray, value: Double) {
    booleanSet(mask, to: value)
  }
}

// MARK: - Fancy (gather) indexing

extension NDArray {

  /// Gather elements at the given flat indices, returning a 1-D array.
  ///
  /// Negative indices are supported (wrap from the end). The result dtype matches
  /// `self`. Indices may repeat or appear in any order.
  ///
  /// - Parameter indices: Flat indices of elements to gather.
  /// - Returns: A 1-D `NDArray` with `indices.count` elements.
  public subscript(indices indices: [Int]) -> NDArray {
    let normalised = indices.map { normalizeIndex($0, size: size) }
    return gatherElements(at: normalised)
  }

  /// Gather elements using a 1-D `.int64` NDArray as the index list.
  ///
  /// Equivalent to `a[indices: idx.int64Data!.map(Int.init)]`, but avoids the
  /// conversion at the call site. Negative indices are supported.
  ///
  /// - Parameter ndIndices: A 1-D `.int64` NDArray of flat indices.
  /// - Returns: A 1-D `NDArray` with the gathered elements.
  public subscript(ndIndices ndIndices: NDArray) -> NDArray {
    precondition(ndIndices.dtype == .int64, "ndIndices must be .int64")
    guard case .int64(let idxData) = ndIndices.storage else {
      preconditionFailure("ndIndices storage is not .int64")
    }
    let normalised = idxData.map { normalizeIndex(Int($0), size: size) }
    return gatherElements(at: normalised)
  }
}

// MARK: - Gather / scatter helpers

extension NDArray {

  /// Gather elements at the pre-validated (non-negative) flat indices.
  internal func gatherElements(at indices: [Int]) -> NDArray {
    let n = indices.count
    switch storage {
    case .float64(let d):
      return NDArray(shape: [n], storage: .float64(indices.map { d[$0] }))
    case .int64(let d):
      return NDArray(shape: [n], storage: .int64(indices.map { d[$0] }))
    case .bool(let d):
      return NDArray(shape: [n], storage: .bool(indices.map { d[$0] }))
    case .complex128(let r, let im):
      return NDArray(shape: [n], storage: .complex128(
        real: indices.map { r[$0] },
        imag: indices.map { im[$0] }))
    case .date(let d):
      return NDArray(shape: [n], storage: .date(indices.map { d[$0] }))
    }
  }

  /// Assign a `Double` scalar to the given flat indices (mutating).
  private mutating func applyScalar(_ value: Double, at indices: [Int]) {
    switch storage {
    case .float64(var d):
      for i in indices { d[i] = value }
      storage = .float64(d)
    case .date(var d):
      for i in indices { d[i] = value }
      storage = .date(d)
    default:
      var r = real
      for i in indices { r[i] = value }
      storage = .float64(r)
    }
  }
}
