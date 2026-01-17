//
//  NDArrayManipulation.swift
//  ArraySwift
//
//  Shape manipulation and array operations for NDArray
//

import Foundation
import Accelerate

// MARK: - Shape Manipulation

extension NDArray {

    // MARK: - Reshape

    /// Reshape the array to a new shape.
    /// - Parameter newShape: New shape (total size must match)
    public func reshape(_ newShape: [Int]) -> NDArray {
        let newSize = newShape.reduce(1, *)
        precondition(newSize == size, "Cannot reshape array of size \(size) to shape \(newShape)")

        if isComplex {
            return NDArray(shape: newShape, dtype: .complex128, real: real, imag: imag)
        }
        return NDArray(shape: newShape, data: real)
    }

    /// Flatten the array to 1D.
    public func flatten() -> NDArray {
        if isComplex {
            return NDArray(shape: [size], dtype: .complex128, real: real, imag: imag)
        }
        return NDArray(shape: [size], data: real)
    }

    /// Flatten the array to 1D (alias for flatten).
    public func ravel() -> NDArray {
        flatten()
    }

    /// Remove single-dimensional entries from the shape.
    public func squeeze() -> NDArray {
        let newShape = shape.filter { $0 != 1 }
        if newShape.isEmpty {
            return NDArray(shape: [1], data: real)
        }
        return reshape(newShape)
    }

    /// Remove a specific axis if it has size 1.
    public func squeeze(axis: Int) -> NDArray {
        let normalizedAxis = axis < 0 ? ndim + axis : axis
        guard normalizedAxis >= 0 && normalizedAxis < ndim else { return self }
        guard shape[normalizedAxis] == 1 else { return self }

        var newShape = shape
        newShape.remove(at: normalizedAxis)
        if newShape.isEmpty { newShape = [1] }
        return reshape(newShape)
    }

    /// Add a new axis at the specified position.
    public func expandDims(axis: Int) -> NDArray {
        var normalizedAxis = axis < 0 ? ndim + axis + 1 : axis
        normalizedAxis = Swift.max(0, Swift.min(normalizedAxis, ndim))

        var newShape = shape
        newShape.insert(1, at: normalizedAxis)
        return reshape(newShape)
    }

    // MARK: - Transpose

    /// Transpose the array (reverse axes).
    public func transpose() -> NDArray {
        let axes = Array((0..<ndim).reversed())
        return transpose(axes: axes)
    }

    /// Transpose with specified axis order.
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

    /// Swap two axes.
    public func swapaxes(_ axis1: Int, _ axis2: Int) -> NDArray {
        var axes = Array(0..<ndim)
        axes[axis1] = axis2
        axes[axis2] = axis1
        return transpose(axes: axes)
    }

    /// Move an axis to a new position.
    public func moveaxis(source: Int, destination: Int) -> NDArray {
        var axes = Array(0..<ndim)
        let src = source < 0 ? ndim + source : source
        let dst = destination < 0 ? ndim + destination : destination

        let axis = axes.remove(at: src)
        axes.insert(axis, at: dst)
        return transpose(axes: axes)
    }

    // MARK: - Concatenation and Stacking

    /// Concatenate arrays along an axis.
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
                    precondition(arr.shape[d] == first.shape[d],
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

    /// Stack arrays along a new axis.
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

    /// Split array into multiple sub-arrays.
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
                    results.append(NDArray(shape: sliceShape, dtype: .complex128, real: sliceReal, imag: sliceImag))
                } else {
                    results.append(NDArray(shape: sliceShape, data: sliceReal))
                }
            }
            prevIdx = idx
        }

        return results
    }

    // MARK: - Tiling and Repeating

    /// Tile array by repeating it.
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

    /// Repeat elements of array.
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

    /// Reverse order of elements along an axis.
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
                return NDArray(shape: shape, dtype: .complex128, real: Array(real.reversed()), imag: Array(imagPart.reversed()))
            }
            return NDArray(shape: shape, data: Array(real.reversed()))
        }
    }

    /// Roll elements along an axis.
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

    /// Create a copy of the array.
    public func copy() -> NDArray {
        if isComplex {
            return NDArray(shape: shape, dtype: .complex128, real: Array(real), imag: imag.map { Array($0) })
        }
        return NDArray(shape: shape, data: Array(real))
    }
}

// MARK: - Free Functions (NumPy-style API)

/// Reshape array.
public func reshape(_ array: NDArray, _ shape: [Int]) -> NDArray { array.reshape(shape) }

/// Flatten array.
public func flatten(_ array: NDArray) -> NDArray { array.flatten() }

/// Ravel array.
public func ravel(_ array: NDArray) -> NDArray { array.ravel() }

/// Squeeze array.
public func squeeze(_ array: NDArray) -> NDArray { array.squeeze() }

/// Expand dimensions.
public func expandDims(_ array: NDArray, axis: Int) -> NDArray { array.expandDims(axis: axis) }

/// Transpose array.
public func transpose(_ array: NDArray) -> NDArray { array.transpose() }

/// Concatenate arrays.
public func concatenate(_ arrays: [NDArray], axis: Int = 0) -> NDArray { NDArray.concatenate(arrays, axis: axis) }

/// Stack arrays.
public func stack(_ arrays: [NDArray], axis: Int = 0) -> NDArray { NDArray.stack(arrays, axis: axis) }

/// Tile array.
public func tile(_ array: NDArray, _ reps: [Int]) -> NDArray { array.tile(reps) }

/// Flip array.
public func flip(_ array: NDArray, axis: Int? = nil) -> NDArray { array.flip(axis: axis) }

/// Roll array.
public func roll(_ array: NDArray, _ shift: Int, axis: Int? = nil) -> NDArray { array.roll(shift, axis: axis) }
