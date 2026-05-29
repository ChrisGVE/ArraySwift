//
//  NDArrayManipulation.swift
//  ArraySwift
//
//  Shape manipulation and array operations for NDArray
//

import Accelerate
import Foundation

// MARK: - Shape Manipulation

extension NDArray {

  // MARK: - Reshape

  /// Reshape the array to a new shape.
  /// One dimension may be -1, in which case it is inferred from the total size.
  /// - Parameter newShape: New shape (total size must match, at most one -1 allowed)
  public func reshape(_ newShape: [Int]) -> NDArray {
    var resolvedShape = newShape
    let inferIdx = newShape.firstIndex(of: -1)

    if let idx = inferIdx {
      precondition(
        newShape.filter({ $0 == -1 }).count == 1,
        "Only one dimension can be -1 in reshape"
      )
      let knownProduct = newShape.enumerated()
        .filter({ $0.offset != idx })
        .map({ $0.element })
        .reduce(1, *)
      precondition(
        knownProduct > 0 && size % knownProduct == 0,
        "Cannot infer dimension: size \(size) is not divisible by \(knownProduct)")
      resolvedShape[idx] = size / knownProduct
    }

    let newSize = resolvedShape.reduce(1, *)
    precondition(newSize == size, "Cannot reshape array of size \(size) to shape \(resolvedShape)")

    return NDArray(shape: resolvedShape, storage: storage)
  }

  /// Return a 1-D copy of the array. Equivalent to NumPy's `ndarray.flatten`.
  /// - Returns: A 1-D `NDArray` with elements in row-major (C) order.
  public func flatten() -> NDArray {
    return NDArray(shape: [size], storage: storage)
  }

  /// Return a 1-D view (or copy) of the array. Alias for ``flatten()``.
  /// Equivalent to NumPy's `numpy.ravel`.
  /// - Returns: A 1-D `NDArray` with elements in row-major order.
  public func ravel() -> NDArray {
    flatten()
  }

  /// Remove all length-1 dimensions from the shape.
  ///
  /// Equivalent to NumPy's `numpy.squeeze`. If all dimensions are 1, a shape of `[1]`
  /// is returned rather than a scalar.
  ///
  /// - Returns: An `NDArray` with the same data but no size-1 axes.
  public func squeeze() -> NDArray {
    let newShape = shape.filter { $0 != 1 }
    if newShape.isEmpty {
      return NDArray(shape: [1], storage: storage)
    }
    return reshape(newShape)
  }

  /// Remove a specific axis if its length is 1.
  ///
  /// If the axis length is not 1, the array is returned unchanged.
  /// Supports negative axis indexing.
  ///
  /// - Parameter axis: Axis to remove. Must refer to a dimension of size 1.
  /// - Returns: An `NDArray` with the specified axis removed (or `self` if the axis
  ///   length is not 1).
  public func squeeze(axis: Int) -> NDArray {
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else { return self }
    guard shape[normalizedAxis] == 1 else { return self }

    var newShape = shape
    newShape.remove(at: normalizedAxis)
    if newShape.isEmpty { newShape = [1] }
    return reshape(newShape)
  }

  /// Insert a new length-1 axis at the specified position.
  ///
  /// Equivalent to NumPy's `numpy.expand_dims`. Supports negative axis indexing
  /// (e.g. `-1` appends a trailing axis).
  ///
  /// - Parameter axis: Position at which to insert the new axis.
  /// - Returns: An `NDArray` with one additional dimension of size 1.
  public func expandDims(axis: Int) -> NDArray {
    var normalizedAxis = axis < 0 ? ndim + axis + 1 : axis
    normalizedAxis = Swift.max(0, Swift.min(normalizedAxis, ndim))

    var newShape = shape
    newShape.insert(1, at: normalizedAxis)
    return reshape(newShape)
  }

  // MARK: - Transpose

  /// Return the array with all axes reversed. Equivalent to NumPy's `ndarray.transpose()`.
  ///
  /// For a 2-D array this is the standard matrix transpose. Preserves complex dtype.
  ///
  /// - Returns: A new `NDArray` with reversed axis order.
  public func transpose() -> NDArray {
    let axes = Array((0..<ndim).reversed())
    return transpose(axes: axes)
  }

  /// Return the array transposed according to an explicit axis permutation.
  ///
  /// Equivalent to NumPy's `numpy.transpose(a, axes)`. Preserves complex dtype.
  ///
  /// - Parameter axes: A permutation of `0..<ndim` specifying the new axis order.
  /// - Returns: A new `NDArray` with axes permuted as specified.
  public func transpose(axes: [Int]) -> NDArray {
    precondition(axes.count == ndim, "Axes must have same length as dimensions")
    precondition(Set(axes) == Set(0..<ndim), "Axes must be a permutation of 0..<ndim")

    // Calculate new shape
    let newShape = axes.map { shape[$0] }
    let newSize = newShape.reduce(1, *)

    // Calculate strides for old and new arrays
    var oldStrides = [Int](repeating: 1, count: ndim)
    for i in stride(from: ndim - 2, through: 0, by: -1) {
      oldStrides[i] = oldStrides[i + 1] * shape[i + 1]
    }

    var newStrides = [Int](repeating: 1, count: ndim)
    for i in stride(from: ndim - 2, through: 0, by: -1) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1]
    }

    var newReal = [Double](repeating: 0, count: newSize)
    var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: newSize) : nil

    // Reorder data
    for i in 0..<newSize {
      // Convert new flat index to new multi-index
      var newIndices = [Int](repeating: 0, count: ndim)
      var remaining = i
      for d in 0..<ndim {
        newIndices[d] = remaining / newStrides[d]
        remaining = remaining % newStrides[d]
      }

      // Convert to old indices
      var oldFlatIdx = 0
      for d in 0..<ndim {
        oldFlatIdx += newIndices[d] * oldStrides[axes[d]]
      }

      newReal[i] = real[oldFlatIdx]
      if isComplex, let imagPart = imag {
        newImag![i] = imagPart[oldFlatIdx]
      }
    }

    if isComplex {
      return NDArray(shape: newShape, dtype: .complex128, real: newReal, imag: newImag)
    }
    return NDArray(shape: newShape, data: newReal)
  }

  /// Transpose shorthand (equivalent to `transpose()`).
  public var T: NDArray { transpose() }

  /// Interchange two axes. Equivalent to NumPy's `numpy.swapaxes`.
  /// - Parameters:
  ///   - axis1: First axis index (supports negative indexing).
  ///   - axis2: Second axis index (supports negative indexing).
  /// - Returns: A new `NDArray` with the two axes swapped.
  public func swapaxes(_ axis1: Int, _ axis2: Int) -> NDArray {
    var axes = Array(0..<ndim)
    axes[axis1] = axis2
    axes[axis2] = axis1
    return transpose(axes: axes)
  }

  /// Move an axis from `source` to `destination`, shifting other axes accordingly.
  ///
  /// Equivalent to NumPy's `numpy.moveaxis`. Supports negative axis indexing.
  ///
  /// - Parameters:
  ///   - source: Original axis position.
  ///   - destination: Target axis position.
  /// - Returns: A new `NDArray` with the axis moved.
  public func moveaxis(source: Int, destination: Int) -> NDArray {
    var axes = Array(0..<ndim)
    let src = source < 0 ? ndim + source : source
    let dst = destination < 0 ? ndim + destination : destination

    let axis = axes.remove(at: src)
    axes.insert(axis, at: dst)
    return transpose(axes: axes)
  }

  // MARK: - Concatenation and Stacking

  /// Join a sequence of arrays along an existing axis.
  ///
  /// Equivalent to NumPy's `numpy.concatenate`. All arrays must have the same number
  /// of dimensions and identical shapes except along `axis`. If any input is complex,
  /// the output is `.complex128`.
  ///
  /// - Parameters:
  ///   - arrays: Arrays to concatenate. Must be non-empty.
  ///   - axis: Axis along which to concatenate (default `0`). Supports negative indexing.
  /// - Returns: A new `NDArray` whose `axis` dimension is the sum of the input dimensions.
  public static func concatenate(_ arrays: [NDArray], axis: Int = 0) -> NDArray {
    guard !arrays.isEmpty else { return NDArray(shape: [0], data: []) }
    guard arrays.count > 1 else { return arrays[0] }

    let first = arrays[0]
    let normalizedAxis = axis < 0 ? first.ndim + axis : axis

    // Verify shapes match except along concatenation axis
    for arr in arrays.dropFirst() {
      precondition(arr.ndim == first.ndim, "All arrays must have same number of dimensions")
      for d in 0..<first.ndim {
        if d != normalizedAxis {
          precondition(
            arr.shape[d] == first.shape[d],
            "Array shapes must match except along concatenation axis")
        }
      }
    }

    // Check if any array is complex
    let hasComplex = arrays.contains { $0.isComplex }

    // Calculate result shape
    var resultShape = first.shape
    resultShape[normalizedAxis] = arrays.reduce(0) { $0 + $1.shape[normalizedAxis] }

    let resultSize = resultShape.reduce(1, *)
    var resultReal = [Double](repeating: 0, count: resultSize)
    var resultImag: [Double]? = hasComplex ? [Double](repeating: 0, count: resultSize) : nil

    // Calculate strides
    var resultStrides = [Int](repeating: 1, count: first.ndim)
    for i in stride(from: first.ndim - 2, through: 0, by: -1) {
      resultStrides[i] = resultStrides[i + 1] * resultShape[i + 1]
    }

    // Copy data
    var axisOffset = 0
    for arr in arrays {
      var arrStrides = [Int](repeating: 1, count: arr.ndim)
      for i in stride(from: arr.ndim - 2, through: 0, by: -1) {
        arrStrides[i] = arrStrides[i + 1] * arr.shape[i + 1]
      }

      for i in 0..<arr.size {
        // Convert to indices
        var indices = [Int](repeating: 0, count: arr.ndim)
        var remaining = i
        for d in 0..<arr.ndim {
          indices[d] = remaining / arrStrides[d]
          remaining = remaining % arrStrides[d]
        }

        // Offset the concatenation axis
        indices[normalizedAxis] += axisOffset

        // Calculate result index
        var resultIdx = 0
        for d in 0..<first.ndim {
          resultIdx += indices[d] * resultStrides[d]
        }

        resultReal[resultIdx] = arr.real[i]
        if hasComplex {
          resultImag![resultIdx] = arr.imag?[i] ?? 0
        }
      }

      axisOffset += arr.shape[normalizedAxis]
    }

    if hasComplex {
      return NDArray(shape: resultShape, dtype: .complex128, real: resultReal, imag: resultImag)
    }
    return NDArray(shape: resultShape, data: resultReal)
  }

  /// Join a sequence of arrays along a **new** axis.
  ///
  /// Equivalent to NumPy's `numpy.stack`. All arrays must have identical shapes.
  ///
  /// - Parameters:
  ///   - arrays: Arrays to stack. Must be non-empty and all share the same shape.
  ///   - axis: Position of the new axis in the result (default `0`).
  /// - Returns: A new `NDArray` with one additional dimension of size `arrays.count`.
  public static func stack(_ arrays: [NDArray], axis: Int = 0) -> NDArray {
    guard !arrays.isEmpty else { return NDArray(shape: [0], data: []) }

    // Verify all arrays have same shape
    let first = arrays[0]
    for arr in arrays.dropFirst() {
      precondition(arr.shape == first.shape, "All arrays must have same shape for stack")
    }

    // Expand each array and concatenate
    let expanded = arrays.map { $0.expandDims(axis: axis) }
    return concatenate(expanded, axis: axis)
  }

  /// Split the array into multiple sub-arrays at the given indices along an axis.
  ///
  /// Equivalent to NumPy's `numpy.split`. `indices` marks the starting positions of
  /// each slice; the final slice extends to the end of the axis.
  ///
  /// - Parameters:
  ///   - indices: Sorted list of split points along `axis`.
  ///   - axis: Axis along which to split (default `0`). Supports negative indexing.
  /// - Returns: Array of sub-arrays partitioned at the given indices.
  public func split(indices: [Int], axis: Int = 0) -> [NDArray] {
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    var results: [NDArray] = []

    var prevIdx = 0
    for idx in indices + [shape[normalizedAxis]] {
      if idx > prevIdx {
        let sliceSize = idx - prevIdx
        var sliceShape = shape
        sliceShape[normalizedAxis] = sliceSize

        let sliceCount = sliceShape.reduce(1, *)
        var sliceReal = [Double](repeating: 0, count: sliceCount)
        var sliceImag: [Double]? = isComplex ? [Double](repeating: 0, count: sliceCount) : nil

        // Calculate strides
        var strides = [Int](repeating: 1, count: ndim)
        for i in stride(from: ndim - 2, through: 0, by: -1) {
          strides[i] = strides[i + 1] * shape[i + 1]
        }

        var sliceStrides = [Int](repeating: 1, count: ndim)
        for i in stride(from: ndim - 2, through: 0, by: -1) {
          sliceStrides[i] = sliceStrides[i + 1] * sliceShape[i + 1]
        }

        for i in 0..<sliceCount {
          var indices = [Int](repeating: 0, count: ndim)
          var remaining = i
          for d in 0..<ndim {
            indices[d] = remaining / sliceStrides[d]
            remaining = remaining % sliceStrides[d]
          }

          indices[normalizedAxis] += prevIdx

          var srcIdx = 0
          for d in 0..<ndim {
            srcIdx += indices[d] * strides[d]
          }

          sliceReal[i] = real[srcIdx]
          if isComplex, let imagPart = imag {
            sliceImag![i] = imagPart[srcIdx]
          }
        }

        if isComplex {
          results.append(
            NDArray(shape: sliceShape, dtype: .complex128, real: sliceReal, imag: sliceImag))
        } else {
          results.append(NDArray(shape: sliceShape, data: sliceReal))
        }
      }
      prevIdx = idx
    }

    return results
  }

  // MARK: - Tiling and Repeating

  /// Construct an array by repeating `self` the number of times given by `reps`.
  ///
  /// Equivalent to NumPy's `numpy.tile`. When `reps` is shorter than `ndim`, it is
  /// left-padded with ones. When it is longer, the array is expanded with leading
  /// size-1 dimensions first.
  ///
  /// - Parameter reps: Number of times to repeat along each axis.
  /// - Returns: A new tiled `NDArray`.
  public func tile(_ reps: [Int]) -> NDArray {
    var result = self

    // Pad reps or shape to match
    var paddedReps = reps
    while paddedReps.count < result.ndim {
      paddedReps.insert(1, at: 0)
    }
    while result.ndim < paddedReps.count {
      result = result.expandDims(axis: 0)
    }

    // Tile along each axis
    for axis in 0..<paddedReps.count {
      if paddedReps[axis] > 1 {
        var arrays: [NDArray] = []
        for _ in 0..<paddedReps[axis] {
          arrays.append(result)
        }
        result = NDArray.concatenate(arrays, axis: axis)
      }
    }

    return result
  }

  /// Repeat each element a fixed number of times.
  ///
  /// Equivalent to NumPy's `numpy.repeat`.
  ///
  /// - When `axis` is `nil`: flattens the array first, then repeats every element.
  /// - When `axis` is given: repeats along that axis without flattening.
  ///
  /// - Parameters:
  ///   - repeats: Number of times to repeat each element.
  ///   - axis: Axis along which to repeat. Pass `nil` (default) to repeat on a
  ///     flattened view.
  /// - Returns: A new `NDArray` with repeated elements.
  public func `repeat`(_ repeats: Int, axis: Int? = nil) -> NDArray {
    if let ax = axis {
      let normalizedAxis = ax < 0 ? ndim + ax : ax
      var newShape = shape
      newShape[normalizedAxis] *= repeats

      let newSize = newShape.reduce(1, *)
      var newReal = [Double](repeating: 0, count: newSize)
      var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: newSize) : nil

      // Calculate strides
      var oldStrides = [Int](repeating: 1, count: ndim)
      for i in stride(from: ndim - 2, through: 0, by: -1) {
        oldStrides[i] = oldStrides[i + 1] * shape[i + 1]
      }

      var newStrides = [Int](repeating: 1, count: ndim)
      for i in stride(from: ndim - 2, through: 0, by: -1) {
        newStrides[i] = newStrides[i + 1] * newShape[i + 1]
      }

      for i in 0..<size {
        var indices = [Int](repeating: 0, count: ndim)
        var remaining = i
        for d in 0..<ndim {
          indices[d] = remaining / oldStrides[d]
          remaining = remaining % oldStrides[d]
        }

        for r in 0..<repeats {
          var newIndices = indices
          newIndices[normalizedAxis] = indices[normalizedAxis] * repeats + r

          var newIdx = 0
          for d in 0..<ndim {
            newIdx += newIndices[d] * newStrides[d]
          }

          newReal[newIdx] = real[i]
          if isComplex, let imagPart = imag {
            newImag![newIdx] = imagPart[i]
          }
        }
      }

      if isComplex {
        return NDArray(shape: newShape, dtype: .complex128, real: newReal, imag: newImag)
      }
      return NDArray(shape: newShape, data: newReal)
    } else {
      // Repeat all elements (flatten first)
      let newSize = size * repeats
      var newReal = [Double](repeating: 0, count: newSize)
      var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: newSize) : nil

      for i in 0..<size {
        for r in 0..<repeats {
          newReal[i * repeats + r] = real[i]
          if isComplex, let imagPart = imag {
            newImag![i * repeats + r] = imagPart[i]
          }
        }
      }

      if isComplex {
        return NDArray(shape: [newSize], dtype: .complex128, real: newReal, imag: newImag)
      }
      return NDArray(shape: [newSize], data: newReal)
    }
  }

  // MARK: - Flip and Roll

  /// Reverse the order of elements along one or all axes.
  ///
  /// Equivalent to NumPy's `numpy.flip`.
  ///
  /// - Parameter axis: Axis to reverse. Pass `nil` (default) to reverse all elements.
  /// - Returns: A new `NDArray` with elements in reversed order.
  public func flip(axis: Int? = nil) -> NDArray {
    if let ax = axis {
      let normalizedAxis = ax < 0 ? ndim + ax : ax

      var strides = [Int](repeating: 1, count: ndim)
      for i in stride(from: ndim - 2, through: 0, by: -1) {
        strides[i] = strides[i + 1] * shape[i + 1]
      }

      var newReal = [Double](repeating: 0, count: size)
      var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: size) : nil

      for i in 0..<size {
        var indices = [Int](repeating: 0, count: ndim)
        var remaining = i
        for d in 0..<ndim {
          indices[d] = remaining / strides[d]
          remaining = remaining % strides[d]
        }

        // Flip the axis
        indices[normalizedAxis] = shape[normalizedAxis] - 1 - indices[normalizedAxis]

        var srcIdx = 0
        for d in 0..<ndim {
          srcIdx += indices[d] * strides[d]
        }

        newReal[i] = real[srcIdx]
        if isComplex, let imagPart = imag {
          newImag![i] = imagPart[srcIdx]
        }
      }

      if isComplex {
        return NDArray(shape: shape, dtype: .complex128, real: newReal, imag: newImag)
      }
      return NDArray(shape: shape, data: newReal)
    } else {
      // Flip all axes
      if isComplex, let imagPart = imag {
        return NDArray(
          shape: shape, dtype: .complex128, real: Array(real.reversed()),
          imag: Array(imagPart.reversed()))
      }
      return NDArray(shape: shape, data: Array(real.reversed()))
    }
  }

  /// Cyclically shift elements by `shift` positions.
  ///
  /// Equivalent to NumPy's `numpy.roll`.
  ///
  /// - Parameters:
  ///   - shift: Number of positions to shift. Positive values shift toward higher indices.
  ///   - axis: Axis along which to shift. Pass `nil` (default) to roll the flattened array.
  /// - Returns: A new `NDArray` with elements cyclically shifted.
  public func roll(_ shift: Int, axis: Int? = nil) -> NDArray {
    if let ax = axis {
      let normalizedAxis = ax < 0 ? ndim + ax : ax
      let axisSize = shape[normalizedAxis]
      let normalizedShift = ((shift % axisSize) + axisSize) % axisSize

      if normalizedShift == 0 { return self }

      var strides = [Int](repeating: 1, count: ndim)
      for i in stride(from: ndim - 2, through: 0, by: -1) {
        strides[i] = strides[i + 1] * shape[i + 1]
      }

      var newReal = [Double](repeating: 0, count: size)
      var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: size) : nil

      for i in 0..<size {
        var indices = [Int](repeating: 0, count: ndim)
        var remaining = i
        for d in 0..<ndim {
          indices[d] = remaining / strides[d]
          remaining = remaining % strides[d]
        }

        // Roll the axis
        indices[normalizedAxis] = (indices[normalizedAxis] - normalizedShift + axisSize) % axisSize

        var srcIdx = 0
        for d in 0..<ndim {
          srcIdx += indices[d] * strides[d]
        }

        newReal[i] = real[srcIdx]
        if isComplex, let imagPart = imag {
          newImag![i] = imagPart[srcIdx]
        }
      }

      if isComplex {
        return NDArray(shape: shape, dtype: .complex128, real: newReal, imag: newImag)
      }
      return NDArray(shape: shape, data: newReal)
    } else {
      // Roll flat array
      let normalizedShift = ((shift % size) + size) % size
      if normalizedShift == 0 { return self }

      var newReal = [Double](repeating: 0, count: size)
      var newImag: [Double]? = isComplex ? [Double](repeating: 0, count: size) : nil

      for i in 0..<size {
        newReal[(i + normalizedShift) % size] = real[i]
        if isComplex, let imagPart = imag {
          newImag![(i + normalizedShift) % size] = imagPart[i]
        }
      }

      if isComplex {
        return NDArray(shape: shape, dtype: .complex128, real: newReal, imag: newImag)
      }
      return NDArray(shape: shape, data: newReal)
    }
  }

  // MARK: - Copy

  /// Return an independent deep copy of the array.
  ///
  /// Equivalent to NumPy's `ndarray.copy`. Modifications to the copy do not
  /// affect the original.
  ///
  /// - Returns: A new `NDArray` with the same shape, dtype, and element values.
  public func copy() -> NDArray {
    if isComplex {
      return NDArray(
        shape: shape, dtype: .complex128, real: Array(real), imag: imag.map { Array($0) })
    }
    return NDArray(shape: shape, data: Array(real))
  }
}

// MARK: - Free Functions (NumPy-style API)

/// Reshape `array` to a new shape. Equivalent to NumPy's `numpy.reshape`.
/// - Parameters:
///   - array: Source array.
///   - shape: Target shape. One dimension may be `-1` for automatic inference.
/// - Returns: A new `NDArray` with the given shape.
public func reshape(_ array: NDArray, _ shape: [Int]) -> NDArray { array.reshape(shape) }

/// Return a 1-D copy of `array`. Equivalent to NumPy's `numpy.ndarray.flatten`.
/// - Parameter array: Source array.
/// - Returns: A 1-D `NDArray`.
public func flatten(_ array: NDArray) -> NDArray { array.flatten() }

/// Return a 1-D view or copy of `array`. Equivalent to NumPy's `numpy.ravel`.
/// - Parameter array: Source array.
/// - Returns: A 1-D `NDArray`.
public func ravel(_ array: NDArray) -> NDArray { array.ravel() }

/// Remove length-1 dimensions from `array`. Equivalent to NumPy's `numpy.squeeze`.
/// - Parameter array: Source array.
/// - Returns: An `NDArray` with size-1 axes removed.
public func squeeze(_ array: NDArray) -> NDArray { array.squeeze() }

/// Insert a new axis at `axis`. Equivalent to NumPy's `numpy.expand_dims`.
/// - Parameters:
///   - array: Source array.
///   - axis: Position for the new axis.
/// - Returns: An `NDArray` with one additional size-1 dimension.
public func expandDims(_ array: NDArray, axis: Int) -> NDArray { array.expandDims(axis: axis) }

/// Return `array` with all axes reversed. Equivalent to NumPy's `numpy.transpose`.
/// - Parameter array: Source array.
/// - Returns: A transposed `NDArray`.
public func transpose(_ array: NDArray) -> NDArray { array.transpose() }

/// Join arrays along an existing axis. Equivalent to NumPy's `numpy.concatenate`.
/// - Parameters:
///   - arrays: Arrays to join.
///   - axis: Axis along which to concatenate (default `0`).
/// - Returns: A concatenated `NDArray`.
public func concatenate(_ arrays: [NDArray], axis: Int = 0) -> NDArray {
  NDArray.concatenate(arrays, axis: axis)
}

/// Join arrays along a new axis. Equivalent to NumPy's `numpy.stack`.
/// - Parameters:
///   - arrays: Arrays to stack (must share the same shape).
///   - axis: Position of the new axis (default `0`).
/// - Returns: A stacked `NDArray`.
public func stack(_ arrays: [NDArray], axis: Int = 0) -> NDArray {
  NDArray.stack(arrays, axis: axis)
}

/// Construct an array by tiling `array`. Equivalent to NumPy's `numpy.tile`.
/// - Parameters:
///   - array: Source array.
///   - reps: Repetitions along each axis.
/// - Returns: A tiled `NDArray`.
public func tile(_ array: NDArray, _ reps: [Int]) -> NDArray { array.tile(reps) }

/// Reverse elements along an axis. Equivalent to NumPy's `numpy.flip`.
/// - Parameters:
///   - array: Source array.
///   - axis: Axis to flip, or `nil` to flip all elements.
/// - Returns: A flipped `NDArray`.
public func flip(_ array: NDArray, axis: Int? = nil) -> NDArray { array.flip(axis: axis) }

/// Cyclically shift elements. Equivalent to NumPy's `numpy.roll`.
/// - Parameters:
///   - array: Source array.
///   - shift: Number of positions to shift.
///   - axis: Axis to roll along, or `nil` to roll the flattened array.
/// - Returns: A rolled `NDArray`.
public func roll(_ array: NDArray, _ shift: Int, axis: Int? = nil) -> NDArray {
  array.roll(shift, axis: axis)
}
