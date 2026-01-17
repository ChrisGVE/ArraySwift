//
//  NDArray.swift
//  ArraySwift
//
//  Created by Christian C. Berclaz on 2026-01-05.
//  Copyright © 2026 Christian C. Berclaz. All rights reserved.
//
//  Licensed under the MIT License.
//

import Foundation
import Accelerate

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

// MARK: - N-Dimensional Array Structure

/// N-dimensional array representation with split storage for complex support
/// Uses separate real/imag arrays for optimal vDSP vectorization
public struct NDArray: Sendable {
    public var shape: [Int]
    public var dtype: ArrayDType
    public var real: [Double]           // Real part (always present)
    public var imag: [Double]?          // Imaginary part (only for complex128)

    public var ndim: Int { shape.count }
    public var size: Int { real.count }
    public var isComplex: Bool { dtype.isComplex }

    /// Backward compatibility: access real data as 'data'
    public var data: [Double] {
        get { real }
        set { real = newValue }
    }

    /// Calculate strides for row-major (C-style) ordering
    public var strides: [Int] {
        var result = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            result[i] = result[i + 1] * shape[i + 1]
        }
        return result
    }

    /// Convert multi-dimensional index to flat index
    public func flatIndex(_ indices: [Int]) -> Int {
        let strides = self.strides
        var index = 0
        for i in 0..<indices.count {
            index += indices[i] * strides[i]
        }
        return index
    }

    // MARK: - Initializers

    /// Create a real (float64) array - backward compatible initializer
    public init(shape: [Int], data: [Double]) {
        self.shape = shape
        self.dtype = .float64
        self.real = data
        self.imag = nil
    }

    /// Create array with explicit dtype
    public init(shape: [Int], dtype: ArrayDType, real: [Double], imag: [Double]?) {
        self.shape = shape
        self.dtype = dtype
        self.real = real
        self.imag = imag
    }

    // MARK: - Factory Methods

    /// Create a real (float64) array
    public static func realArray(shape: [Int], data: [Double]) -> NDArray {
        NDArray(shape: shape, dtype: .float64, real: data, imag: nil)
    }

    /// Create a complex (complex128) array from split real/imag parts
    public static func complexArray(shape: [Int], real: [Double], imag: [Double]) -> NDArray {
        precondition(real.count == imag.count, "real and imag arrays must have same size")
        return NDArray(shape: shape, dtype: .complex128, real: real, imag: imag)
    }

    /// Create a complex array from interleaved format [r0, i0, r1, i1, ...]
    /// Uses vDSP for SIMD-optimized deinterleaving
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

    /// Convert complex array to interleaved format [r0, i0, r1, i1, ...]
    /// Uses vDSP for SIMD-optimized interleaving
    /// Returns nil for real arrays
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

    /// Get real element at flat index
    public func getReal(at index: Int) -> Double {
        real[index]
    }

    /// Get complex element at flat index as (re, im) tuple
    public func getComplex(at index: Int) -> (re: Double, im: Double) {
        (real[index], imag?[index] ?? 0)
    }

    /// Set real element at flat index
    public mutating func setReal(at index: Int, value: Double) {
        real[index] = value
    }

    /// Set complex element at flat index
    public mutating func setComplex(at index: Int, re: Double, im: Double) {
        real[index] = re
        if imag != nil {
            imag![index] = im
        }
    }

    /// Promote real array to complex (with zero imaginary part)
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

    /// Access element at flat index (1D arrays or linear access)
    public subscript(index: Int) -> Double {
        get { real[index] }
        set { real[index] = newValue }
    }

    /// Access element at 2D indices
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

    /// Access element at 3D indices
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

    /// Access element at N-dimensional indices
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

    /// Access row slice from 2D array
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

    /// Access column slice from 2D array
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

    /// Access range of elements (1D slice)
    public subscript(range: Range<Int>) -> NDArray {
        get {
            let sliceData = Array(real[range])
            if isComplex, let imagPart = imag {
                let sliceImag = Array(imagPart[range])
                return NDArray(shape: [sliceData.count], dtype: .complex128, real: sliceData, imag: sliceImag)
            }
            return NDArray(shape: [sliceData.count], data: sliceData)
        }
    }

    /// Access sub-matrix with row and column ranges
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
                return NDArray(shape: [newRows, newCols], dtype: .complex128, real: resultData, imag: resultImag)
            }
            return NDArray(shape: [newRows, newCols], data: resultData)
        }
    }

    /// Access complex element at flat index, returning tuple
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

    /// Access complex element at 2D indices, returning tuple
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

// MARK: - Type Alias for backward compatibility

/// Backward compatibility alias - ArrayData was the internal name
public typealias ArrayData = NDArray
