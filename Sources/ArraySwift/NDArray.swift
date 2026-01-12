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
}

// MARK: - Type Alias for backward compatibility

/// Backward compatibility alias - ArrayData was the internal name
public typealias ArrayData = NDArray
