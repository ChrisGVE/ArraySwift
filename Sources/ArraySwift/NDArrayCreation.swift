//
//  NDArrayCreation.swift
//  ArraySwift
//
//  Factory functions for creating NDArrays
//

import Foundation
import Accelerate

// MARK: - Factory Functions

extension NDArray {

    // MARK: - Basic Creation

    /// Create an array filled with zeros.
    public static func zeros(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
        let size = shape.reduce(1, *)
        let real = [Double](repeating: 0, count: size)
        let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
        return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
    }

    /// Create an array filled with ones.
    public static func ones(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
        let size = shape.reduce(1, *)
        let real = [Double](repeating: 1, count: size)
        let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
        return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
    }

    /// Create an array filled with a specific value.
    public static func full(_ shape: [Int], value: Double, dtype: ArrayDType = .float64) -> NDArray {
        let size = shape.reduce(1, *)
        let real = [Double](repeating: value, count: size)
        let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: size) : nil
        return NDArray(shape: shape, dtype: dtype, real: real, imag: imag)
    }

    /// Create an array filled with a complex value.
    public static func full(_ shape: [Int], real: Double, imag: Double) -> NDArray {
        let size = shape.reduce(1, *)
        let realData = [Double](repeating: real, count: size)
        let imagData = [Double](repeating: imag, count: size)
        return NDArray(shape: shape, dtype: .complex128, real: realData, imag: imagData)
    }

    /// Create an uninitialized array (filled with zeros in practice).
    public static func empty(_ shape: [Int], dtype: ArrayDType = .float64) -> NDArray {
        zeros(shape, dtype: dtype)
    }

    // MARK: - Range Creation

    /// Create an array with evenly spaced values within a given interval.
    /// - Parameters:
    ///   - start: Start of interval (inclusive)
    ///   - stop: End of interval (exclusive)
    ///   - step: Spacing between values (default 1)
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

    /// Create an array with evenly spaced values over a specified interval.
    /// - Parameters:
    ///   - start: Start of interval
    ///   - stop: End of interval
    ///   - num: Number of samples to generate (default 50)
    ///   - endpoint: If true, stop is the last sample (default true)
    public static func linspace(start: Double, stop: Double, num: Int = 50, endpoint: Bool = true) -> NDArray {
        guard num > 0 else { return NDArray(shape: [0], data: []) }
        guard num > 1 else { return NDArray(shape: [1], data: [start]) }

        var data = [Double](repeating: 0, count: num)
        let divisor = endpoint ? Double(num - 1) : Double(num)

        for i in 0..<num {
            data[i] = start + (stop - start) * Double(i) / divisor
        }

        return NDArray(shape: [num], data: data)
    }

    /// Create an array with values spaced evenly on a log scale.
    /// - Parameters:
    ///   - start: Start of interval (10^start)
    ///   - stop: End of interval (10^stop)
    ///   - num: Number of samples (default 50)
    ///   - base: Base of the log scale (default 10)
    public static func logspace(start: Double, stop: Double, num: Int = 50, base: Double = 10) -> NDArray {
        let linear = linspace(start: start, stop: stop, num: num)
        var data = linear.real
        for i in 0..<data.count {
            data[i] = pow(base, data[i])
        }
        return NDArray(shape: [num], data: data)
    }

    /// Create an array with values spaced evenly on a log scale (geometric spacing).
    /// - Parameters:
    ///   - start: Start value (must be positive)
    ///   - stop: End value (must be positive)
    ///   - num: Number of samples (default 50)
    public static func geomspace(start: Double, stop: Double, num: Int = 50) -> NDArray {
        guard start > 0 && stop > 0 else {
            return NDArray(shape: [0], data: [])
        }
        return logspace(start: Darwin.log10(start), stop: Darwin.log10(stop), num: num)
    }

    // MARK: - Identity and Diagonal

    /// Create an identity matrix.
    /// - Parameters:
    ///   - n: Size of the matrix (n x n)
    ///   - dtype: Data type of the array (default: float64)
    public static func eye(_ n: Int, dtype: ArrayDType = .float64) -> NDArray {
        var data = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            data[i * n + i] = 1
        }
        let imag: [Double]? = dtype == .complex128 ? [Double](repeating: 0, count: n * n) : nil
        return NDArray(shape: [n, n], dtype: dtype, real: data, imag: imag)
    }

    /// Create an identity matrix (alias for eye).
    public static func identity(_ n: Int, dtype: ArrayDType = .float64) -> NDArray {
        eye(n, dtype: dtype)
    }

    /// Create a diagonal matrix or extract diagonal from matrix.
    /// - Parameters:
    ///   - values: If 1D, creates diagonal matrix. If 2D, extracts diagonal.
    ///   - k: Diagonal offset (0 = main diagonal, positive = above, negative = below)
    public static func diag(_ values: NDArray, k: Int = 0) -> NDArray {
        if values.ndim == 1 {
            // Create diagonal matrix from 1D array
            let n = values.size + Swift.abs(k)
            var data = [Double](repeating: 0, count: n * n)

            for i in 0..<values.size {
                let row = k >= 0 ? i : i - k
                let col = k >= 0 ? i + k : i
                if row < n && col < n {
                    data[row * n + col] = values.real[i]
                }
            }

            return NDArray(shape: [n, n], data: data)
        } else if values.ndim == 2 {
            // Extract diagonal from 2D matrix
            let rows = values.shape[0]
            let cols = values.shape[1]

            var diagValues: [Double] = []
            let startRow = k >= 0 ? 0 : -k
            let startCol = k >= 0 ? k : 0

            var row = startRow
            var col = startCol
            while row < rows && col < cols {
                diagValues.append(values.real[row * cols + col])
                row += 1
                col += 1
            }

            return NDArray(shape: [diagValues.count], data: diagValues)
        }

        return values
    }

    // MARK: - Random Arrays

    /// Create an array with random values uniformly distributed in [0, 1).
    public static func random(_ shape: [Int]) -> NDArray {
        let size = shape.reduce(1, *)
        var data = [Double](repeating: 0, count: size)
        for i in 0..<size {
            data[i] = Double.random(in: 0..<1)
        }
        return NDArray(shape: shape, data: data)
    }

    /// Create an array with random values from standard normal distribution.
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

    /// Create an array with random integers in [low, high).
    public static func randint(low: Int, high: Int, shape: [Int]) -> NDArray {
        let size = shape.reduce(1, *)
        var data = [Double](repeating: 0, count: size)
        for i in 0..<size {
            data[i] = Double(Int.random(in: low..<high))
        }
        return NDArray(shape: shape, data: data)
    }

    // MARK: - Like Functions

    /// Create an array of zeros with the same shape as another array.
    public static func zerosLike(_ other: NDArray) -> NDArray {
        zeros(other.shape, dtype: other.dtype)
    }

    /// Create an array of ones with the same shape as another array.
    public static func onesLike(_ other: NDArray) -> NDArray {
        ones(other.shape, dtype: other.dtype)
    }

    /// Create an empty array with the same shape as another array.
    public static func emptyLike(_ other: NDArray) -> NDArray {
        empty(other.shape, dtype: other.dtype)
    }

    /// Create a full array with the same shape as another array.
    public static func fullLike(_ other: NDArray, value: Double) -> NDArray {
        full(other.shape, value: value, dtype: other.dtype)
    }

    // MARK: - Complex Array Creation

    /// Create a complex array from separate real and imaginary arrays.
    public static func complexArray(real: NDArray, imag: NDArray) -> NDArray {
        precondition(real.shape == imag.shape, "real and imag must have same shape")
        return NDArray(shape: real.shape, dtype: .complex128, real: real.real, imag: imag.real)
    }

    /// Create a complex array from polar coordinates (magnitude and phase).
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
    /// - Parameters:
    ///   - x: 1D array of x coordinates
    ///   - y: 1D array of y coordinates
    /// - Returns: Tuple of (X, Y) meshgrid arrays
    public static func meshgrid(x: NDArray, y: NDArray) -> (NDArray, NDArray) {
        let nx = x.size
        let ny = y.size

        var xData = [Double](repeating: 0, count: ny * nx)
        var yData = [Double](repeating: 0, count: ny * nx)

        for j in 0..<ny {
            for i in 0..<nx {
                xData[j * nx + i] = x.real[i]
                yData[j * nx + i] = y.real[j]
            }
        }

        let X = NDArray(shape: [ny, nx], data: xData)
        let Y = NDArray(shape: [ny, nx], data: yData)
        return (X, Y)
    }
}

// MARK: - Convenience Initializers

extension NDArray {

    /// Create a 1D array from a Swift array.
    public init(_ data: [Double]) {
        self.init(shape: [data.count], data: data)
    }

    /// Create a 2D array from nested Swift arrays.
    public init(_ data: [[Double]]) {
        let rows = data.count
        let cols = data.first?.count ?? 0
        var flat: [Double] = []
        flat.reserveCapacity(rows * cols)
        for row in data {
            flat.append(contentsOf: row)
        }
        self.init(shape: [rows, cols], data: flat)
    }

    /// Create a 3D array from nested Swift arrays.
    public init(_ data: [[[Double]]]) {
        let d0 = data.count
        let d1 = data.first?.count ?? 0
        let d2 = data.first?.first?.count ?? 0
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
