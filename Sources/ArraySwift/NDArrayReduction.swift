//
//  NDArrayReduction.swift
//  ArraySwift
//
//  Reduction operations for NDArray using vDSP
//

import Foundation
import Accelerate

// MARK: - Reduction Operations

extension NDArray {

    // MARK: - Sum and Product

    /// Sum of all elements.
    public func sum() -> Double {
        var result: Double = 0
        vDSP_sveD(real, 1, &result, vDSP_Length(size))
        return result
    }

    /// Sum along an axis.
    public func sum(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            var result: Double = 0
            vDSP_sveD(slice, 1, &result, vDSP_Length(slice.count))
            return result
        }
    }

    /// Product of all elements.
    public func prod() -> Double {
        guard size > 0 else { return 1 }
        var result = real[0]
        for i in 1..<size {
            result *= real[i]
        }
        return result
    }

    /// Product along an axis.
    public func prod(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            var result = 1.0
            for val in slice {
                result *= val
            }
            return result
        }
    }

    // MARK: - Mean and Variance

    /// Mean of all elements.
    public func mean() -> Double {
        var result: Double = 0
        vDSP_meanvD(real, 1, &result, vDSP_Length(size))
        return result
    }

    /// Mean along an axis.
    public func mean(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            var result: Double = 0
            vDSP_meanvD(slice, 1, &result, vDSP_Length(slice.count))
            return result
        }
    }

    /// Variance of all elements.
    /// - Parameter ddof: Delta degrees of freedom (default 0 for population variance)
    public func variance(ddof: Int = 0) -> Double {
        let m = mean()
        var sumSq: Double = 0
        for i in 0..<size {
            let diff = real[i] - m
            sumSq += diff * diff
        }
        return sumSq / Double(size - ddof)
    }

    /// Variance along an axis.
    public func variance(axis: Int, ddof: Int = 0) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            let m = slice.reduce(0, +) / Double(slice.count)
            var sumSq: Double = 0
            for val in slice {
                let diff = val - m
                sumSq += diff * diff
            }
            return sumSq / Double(slice.count - ddof)
        }
    }

    /// Standard deviation of all elements.
    /// - Parameter ddof: Delta degrees of freedom (default 0 for population std)
    public func std(ddof: Int = 0) -> Double {
        Darwin.sqrt(variance(ddof: ddof))
    }

    /// Standard deviation along an axis.
    public func std(axis: Int, ddof: Int = 0) -> NDArray {
        let v = variance(axis: axis, ddof: ddof)
        return v.sqrt()
    }

    // MARK: - Min and Max

    /// Minimum value.
    public func min() -> Double {
        var result: Double = 0
        vDSP_minvD(real, 1, &result, vDSP_Length(size))
        return result
    }

    /// Minimum along an axis.
    public func min(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            var result: Double = 0
            vDSP_minvD(slice, 1, &result, vDSP_Length(slice.count))
            return result
        }
    }

    /// Maximum value.
    public func max() -> Double {
        var result: Double = 0
        vDSP_maxvD(real, 1, &result, vDSP_Length(size))
        return result
    }

    /// Maximum along an axis.
    public func max(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            var result: Double = 0
            vDSP_maxvD(slice, 1, &result, vDSP_Length(slice.count))
            return result
        }
    }

    /// Index of minimum value.
    public func argmin() -> Int {
        var result: Double = 0
        var idx: vDSP_Length = 0
        vDSP_minviD(real, 1, &result, &idx, vDSP_Length(size))
        return Int(idx)
    }

    /// Index of maximum value.
    public func argmax() -> Int {
        var result: Double = 0
        var idx: vDSP_Length = 0
        vDSP_maxviD(real, 1, &result, &idx, vDSP_Length(size))
        return Int(idx)
    }

    /// Peak-to-peak (max - min) value.
    public func ptp() -> Double {
        max() - min()
    }

    // MARK: - Cumulative Operations

    /// Cumulative sum.
    public func cumsum() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var sum: Double = 0
        for i in 0..<size {
            sum += real[i]
            result[i] = sum
        }
        return NDArray(shape: [size], data: result)
    }

    /// Cumulative product.
    public func cumprod() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var prod: Double = 1
        for i in 0..<size {
            prod *= real[i]
            result[i] = prod
        }
        return NDArray(shape: [size], data: result)
    }

    /// Differences between consecutive elements.
    public func diff(n: Int = 1) -> NDArray {
        guard size > n else { return NDArray(shape: [0], data: []) }

        var current = real
        for _ in 0..<n {
            var newData = [Double](repeating: 0, count: current.count - 1)
            for i in 0..<newData.count {
                newData[i] = current[i + 1] - current[i]
            }
            current = newData
        }

        return NDArray(shape: [current.count], data: current)
    }

    // MARK: - Boolean Reductions

    /// True if all elements are non-zero.
    public func all() -> Bool {
        for i in 0..<size {
            if real[i] == 0 { return false }
        }
        return true
    }

    /// True if all elements along axis are non-zero.
    public func all(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            for val in slice {
                if val == 0 { return 0 }
            }
            return 1
        }
    }

    /// True if any element is non-zero.
    public func any() -> Bool {
        for i in 0..<size {
            if real[i] != 0 { return true }
        }
        return false
    }

    /// True if any element along axis is non-zero.
    public func any(axis: Int) -> NDArray {
        reduceAlongAxis(axis: axis) { slice in
            for val in slice {
                if val != 0 { return 1 }
            }
            return 0
        }
    }

    // MARK: - Statistics

    /// Median value.
    public func median() -> Double {
        let sorted = real.sorted()
        let n = sorted.count
        if n == 0 { return .nan }
        if n % 2 == 0 {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2
        } else {
            return sorted[n/2]
        }
    }

    /// Percentile value.
    /// - Parameter q: Percentile (0-100)
    public func percentile(_ q: Double) -> Double {
        guard size > 0 else { return .nan }
        let sorted = real.sorted()
        let idx = (q / 100) * Double(size - 1)
        let lower = Int(idx)
        let upper = Swift.min(lower + 1, size - 1)
        let frac = idx - Double(lower)
        return sorted[lower] * (1 - frac) + sorted[upper] * frac
    }

    /// Quantile value.
    /// - Parameter q: Quantile (0-1)
    public func quantile(_ q: Double) -> Double {
        percentile(q * 100)
    }

    // MARK: - Axis Reduction Helper

    /// Apply a reduction function along an axis.
    private func reduceAlongAxis(axis: Int, reducer: ([Double]) -> Double) -> NDArray {
        let normalizedAxis = axis < 0 ? ndim + axis : axis
        guard normalizedAxis >= 0 && normalizedAxis < ndim else {
            fatalError("Axis \(axis) out of bounds for array with \(ndim) dimensions")
        }

        // Calculate result shape (remove the reduced axis)
        var resultShape = shape
        resultShape.remove(at: normalizedAxis)
        if resultShape.isEmpty { resultShape = [1] }

        let resultSize = resultShape.reduce(1, *)
        var resultData = [Double](repeating: 0, count: resultSize)

        // Calculate strides
        var strides = [Int](repeating: 1, count: ndim)
        for i in stride(from: ndim - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }

        let axisSize = shape[normalizedAxis]
        let axisStride = strides[normalizedAxis]

        // Iterate over all positions in the result
        for resultIdx in 0..<resultSize {
            // Convert result index to source indices (excluding the reduced axis)
            var sourceIndices = [Int](repeating: 0, count: ndim)
            var remaining = resultIdx
            var resultDim = 0
            for d in 0..<ndim {
                if d == normalizedAxis {
                    sourceIndices[d] = 0  // Will iterate over this
                } else {
                    var resultStride = 1
                    for rd in (resultDim + 1)..<resultShape.count {
                        resultStride *= resultShape[rd]
                    }
                    sourceIndices[d] = remaining / resultStride
                    remaining = remaining % resultStride
                    resultDim += 1
                }
            }

            // Collect slice along the axis
            var slice = [Double](repeating: 0, count: axisSize)
            var baseIdx = 0
            for d in 0..<ndim {
                if d != normalizedAxis {
                    baseIdx += sourceIndices[d] * strides[d]
                }
            }

            for i in 0..<axisSize {
                slice[i] = real[baseIdx + i * axisStride]
            }

            resultData[resultIdx] = reducer(slice)
        }

        return NDArray(shape: resultShape, data: resultData)
    }
}

// MARK: - Free Functions (NumPy-style API)

/// Sum of array elements.
public func sum(_ array: NDArray) -> Double { array.sum() }

/// Sum along axis.
public func sum(_ array: NDArray, axis: Int) -> NDArray { array.sum(axis: axis) }

/// Product of array elements.
public func prod(_ array: NDArray) -> Double { array.prod() }

/// Product along axis.
public func prod(_ array: NDArray, axis: Int) -> NDArray { array.prod(axis: axis) }

/// Mean of array elements.
public func mean(_ array: NDArray) -> Double { array.mean() }

/// Mean along axis.
public func mean(_ array: NDArray, axis: Int) -> NDArray { array.mean(axis: axis) }

/// Variance of array elements.
public func variance(_ array: NDArray, ddof: Int = 0) -> Double { array.variance(ddof: ddof) }

/// Standard deviation of array elements.
public func std(_ array: NDArray, ddof: Int = 0) -> Double { array.std(ddof: ddof) }

/// Minimum of array elements.
public func min(_ array: NDArray) -> Double { array.min() }

/// Maximum of array elements.
public func max(_ array: NDArray) -> Double { array.max() }

/// Index of minimum.
public func argmin(_ array: NDArray) -> Int { array.argmin() }

/// Index of maximum.
public func argmax(_ array: NDArray) -> Int { array.argmax() }

/// Cumulative sum.
public func cumsum(_ array: NDArray) -> NDArray { array.cumsum() }

/// Cumulative product.
public func cumprod(_ array: NDArray) -> NDArray { array.cumprod() }

/// Differences between consecutive elements.
public func diff(_ array: NDArray, n: Int = 1) -> NDArray { array.diff(n: n) }

/// True if all elements are non-zero.
public func all(_ array: NDArray) -> Bool { array.all() }

/// True if any element is non-zero.
public func any(_ array: NDArray) -> Bool { array.any() }

/// Median value.
public func median(_ array: NDArray) -> Double { array.median() }

/// Percentile value.
public func percentile(_ array: NDArray, _ q: Double) -> Double { array.percentile(q) }

/// Quantile value.
public func quantile(_ array: NDArray, _ q: Double) -> Double { array.quantile(q) }
