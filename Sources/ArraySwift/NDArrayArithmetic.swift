//
//  NDArrayArithmetic.swift
//  ArraySwift
//
//  Arithmetic operations for NDArray using vDSP
//

import Foundation
import Accelerate

// MARK: - Arithmetic Operations

extension NDArray {

    // MARK: - Negation

    /// Negate all elements.
    public func negated() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        vDSP_vnegD(real, 1, &result, 1, vDSP_Length(size))

        if isComplex, let imagPart = imag {
            var resultImag = [Double](repeating: 0, count: size)
            vDSP_vnegD(imagPart, 1, &resultImag, 1, vDSP_Length(size))
            return NDArray(shape: shape, dtype: .complex128, real: result, imag: resultImag)
        }

        return NDArray(shape: shape, data: result)
    }

    // MARK: - Element-wise Binary Operations

    /// Add two arrays element-wise.
    public func add(_ other: NDArray) -> NDArray {
        if isComplex || other.isComplex {
            return complexAdd(self.promoteToComplex(), other.promoteToComplex())
        }

        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        vDSP_vaddD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
        return NDArray(shape: resultShape, data: result)
    }

    /// Subtract two arrays element-wise.
    public func subtract(_ other: NDArray) -> NDArray {
        if isComplex || other.isComplex {
            return complexSub(self.promoteToComplex(), other.promoteToComplex())
        }

        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        // vDSP_vsubD computes B - A
        vDSP_vsubD(b.real, 1, a.real, 1, &result, 1, vDSP_Length(a.size))
        return NDArray(shape: resultShape, data: result)
    }

    /// Multiply two arrays element-wise.
    public func multiply(_ other: NDArray) -> NDArray {
        if isComplex || other.isComplex {
            return complexMul(self.promoteToComplex(), other.promoteToComplex())
        }

        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        vDSP_vmulD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
        return NDArray(shape: resultShape, data: result)
    }

    /// Divide two arrays element-wise.
    public func divide(_ other: NDArray) -> NDArray {
        if isComplex || other.isComplex {
            return complexDiv(self.promoteToComplex(), other.promoteToComplex())
        }

        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        // vDSP_vdivD computes B / A
        vDSP_vdivD(b.real, 1, a.real, 1, &result, 1, vDSP_Length(a.size))
        return NDArray(shape: resultShape, data: result)
    }

    /// Raise to power element-wise.
    public func power(_ exponent: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = pow(real[i], exponent)
        }
        return NDArray(shape: shape, data: result)
    }

    /// Raise to power element-wise (array exponent).
    public func power(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = pow(a.real[i], b.real[i])
        }
        return NDArray(shape: resultShape, data: result)
    }

    // MARK: - Scalar Operations

    /// Add a scalar to all elements.
    public func add(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var s = scalar
        vDSP_vsaddD(real, 1, &s, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    /// Subtract a scalar from all elements.
    public func subtract(_ scalar: Double) -> NDArray {
        add(-scalar)
    }

    /// Multiply all elements by a scalar.
    public func multiply(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var s = scalar
        vDSP_vsmulD(real, 1, &s, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    /// Divide all elements by a scalar.
    public func divide(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var s = scalar
        vDSP_vsdivD(real, 1, &s, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    /// Divide scalar by all elements.
    public func scalarDivide(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var s = scalar
        vDSP_svdivD(&s, real, 1, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Modulo Operations

    /// Element-wise modulo operation.
    public func mod(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i].truncatingRemainder(dividingBy: b.real[i])
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise floor modulo (Python-style %).
    public func floorMod(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            let remainder = a.real[i].truncatingRemainder(dividingBy: b.real[i])
            // Adjust for Python-style modulo
            if remainder != 0 && (remainder < 0) != (b.real[i] < 0) {
                result[i] = remainder + b.real[i]
            } else {
                result[i] = remainder
            }
        }
        return NDArray(shape: resultShape, data: result)
    }
}

// MARK: - Complex Arithmetic Helpers

private func complexAdd(_ a: NDArray, _ b: NDArray) -> NDArray {
    let size = a.size
    var resultReal = [Double](repeating: 0, count: size)
    var resultImag = [Double](repeating: 0, count: size)

    a.real.withUnsafeBufferPointer { aRealBuf in
        a.imag!.withUnsafeBufferPointer { aImagBuf in
            b.real.withUnsafeBufferPointer { bRealBuf in
                b.imag!.withUnsafeBufferPointer { bImagBuf in
                    resultReal.withUnsafeMutableBufferPointer { resRealBuf in
                        resultImag.withUnsafeMutableBufferPointer { resImagBuf in
                            var splitA = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: aRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: aImagBuf.baseAddress!)
                            )
                            var splitB = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: bRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: bImagBuf.baseAddress!)
                            )
                            var splitC = DSPDoubleSplitComplex(
                                realp: resRealBuf.baseAddress!,
                                imagp: resImagBuf.baseAddress!
                            )
                            vDSP_zvaddD(&splitA, 1, &splitB, 1, &splitC, 1, vDSP_Length(size))
                        }
                    }
                }
            }
        }
    }

    return NDArray.complexArray(shape: a.shape, real: resultReal, imag: resultImag)
}

private func complexSub(_ a: NDArray, _ b: NDArray) -> NDArray {
    let size = a.size
    var resultReal = [Double](repeating: 0, count: size)
    var resultImag = [Double](repeating: 0, count: size)

    a.real.withUnsafeBufferPointer { aRealBuf in
        a.imag!.withUnsafeBufferPointer { aImagBuf in
            b.real.withUnsafeBufferPointer { bRealBuf in
                b.imag!.withUnsafeBufferPointer { bImagBuf in
                    resultReal.withUnsafeMutableBufferPointer { resRealBuf in
                        resultImag.withUnsafeMutableBufferPointer { resImagBuf in
                            var splitA = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: aRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: aImagBuf.baseAddress!)
                            )
                            var splitB = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: bRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: bImagBuf.baseAddress!)
                            )
                            var splitC = DSPDoubleSplitComplex(
                                realp: resRealBuf.baseAddress!,
                                imagp: resImagBuf.baseAddress!
                            )
                            vDSP_zvsubD(&splitA, 1, &splitB, 1, &splitC, 1, vDSP_Length(size))
                        }
                    }
                }
            }
        }
    }

    return NDArray.complexArray(shape: a.shape, real: resultReal, imag: resultImag)
}

private func complexMul(_ a: NDArray, _ b: NDArray) -> NDArray {
    let size = a.size
    var resultReal = [Double](repeating: 0, count: size)
    var resultImag = [Double](repeating: 0, count: size)

    a.real.withUnsafeBufferPointer { aRealBuf in
        a.imag!.withUnsafeBufferPointer { aImagBuf in
            b.real.withUnsafeBufferPointer { bRealBuf in
                b.imag!.withUnsafeBufferPointer { bImagBuf in
                    resultReal.withUnsafeMutableBufferPointer { resRealBuf in
                        resultImag.withUnsafeMutableBufferPointer { resImagBuf in
                            var splitA = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: aRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: aImagBuf.baseAddress!)
                            )
                            var splitB = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: bRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: bImagBuf.baseAddress!)
                            )
                            var splitC = DSPDoubleSplitComplex(
                                realp: resRealBuf.baseAddress!,
                                imagp: resImagBuf.baseAddress!
                            )
                            vDSP_zvmulD(&splitA, 1, &splitB, 1, &splitC, 1, vDSP_Length(size), 1)
                        }
                    }
                }
            }
        }
    }

    return NDArray.complexArray(shape: a.shape, real: resultReal, imag: resultImag)
}

private func complexDiv(_ a: NDArray, _ b: NDArray) -> NDArray {
    let size = a.size
    var resultReal = [Double](repeating: 0, count: size)
    var resultImag = [Double](repeating: 0, count: size)

    a.real.withUnsafeBufferPointer { aRealBuf in
        a.imag!.withUnsafeBufferPointer { aImagBuf in
            b.real.withUnsafeBufferPointer { bRealBuf in
                b.imag!.withUnsafeBufferPointer { bImagBuf in
                    resultReal.withUnsafeMutableBufferPointer { resRealBuf in
                        resultImag.withUnsafeMutableBufferPointer { resImagBuf in
                            var splitA = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: aRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: aImagBuf.baseAddress!)
                            )
                            var splitB = DSPDoubleSplitComplex(
                                realp: UnsafeMutablePointer(mutating: bRealBuf.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: bImagBuf.baseAddress!)
                            )
                            var splitC = DSPDoubleSplitComplex(
                                realp: resRealBuf.baseAddress!,
                                imagp: resImagBuf.baseAddress!
                            )
                            // vDSP_zvdivD computes B / A
                            vDSP_zvdivD(&splitB, 1, &splitA, 1, &splitC, 1, vDSP_Length(size))
                        }
                    }
                }
            }
        }
    }

    return NDArray.complexArray(shape: a.shape, real: resultReal, imag: resultImag)
}

// MARK: - Broadcasting Helper

/// Broadcast two arrays to compatible shapes.
private func broadcast(_ a: NDArray, _ b: NDArray) -> (NDArray, NDArray, [Int]) {
    if a.shape == b.shape {
        return (a, b, a.shape)
    }

    // Calculate broadcast shape
    let maxDim = max(a.ndim, b.ndim)
    let shapeA = [Int](repeating: 1, count: maxDim - a.ndim) + a.shape
    let shapeB = [Int](repeating: 1, count: maxDim - b.ndim) + b.shape
    var resultShape = [Int](repeating: 0, count: maxDim)

    for i in 0..<maxDim {
        if shapeA[i] == shapeB[i] {
            resultShape[i] = shapeA[i]
        } else if shapeA[i] == 1 {
            resultShape[i] = shapeB[i]
        } else if shapeB[i] == 1 {
            resultShape[i] = shapeA[i]
        } else {
            fatalError("Cannot broadcast shapes \(a.shape) and \(b.shape)")
        }
    }

    let broadcastA = broadcastTo(a, shape: resultShape)
    let broadcastB = broadcastTo(b, shape: resultShape)

    return (broadcastA, broadcastB, resultShape)
}

/// Broadcast an array to a new shape.
private func broadcastTo(_ array: NDArray, shape: [Int]) -> NDArray {
    if array.shape == shape { return array }

    let resultSize = shape.reduce(1, *)
    var result = [Double](repeating: 0, count: resultSize)

    // Pad array shape with leading 1s
    let paddedShape = [Int](repeating: 1, count: shape.count - array.ndim) + array.shape

    // Calculate strides for the result
    var resultStrides = [Int](repeating: 1, count: shape.count)
    for i in stride(from: shape.count - 2, through: 0, by: -1) {
        resultStrides[i] = resultStrides[i + 1] * shape[i + 1]
    }

    // Calculate strides for the source (0 for broadcast dimensions)
    var sourceStrides = [Int](repeating: 0, count: shape.count)
    var strideValue = 1
    for i in Swift.stride(from: array.ndim - 1, through: 0, by: -1) {
        let paddedIdx = i + (shape.count - array.ndim)
        if array.shape[i] == shape[paddedIdx] {
            sourceStrides[paddedIdx] = strideValue
            strideValue *= array.shape[i]
        }
    }

    // Fill result array
    for i in 0..<resultSize {
        var sourceIdx = 0
        var remaining = i
        for d in 0..<shape.count {
            let coord = remaining / resultStrides[d]
            remaining = remaining % resultStrides[d]
            sourceIdx += (coord % paddedShape[d]) * sourceStrides[d]
        }
        result[i] = array.real[sourceIdx]
    }

    return NDArray(shape: shape, data: result)
}

// MARK: - Operators

extension NDArray {
    public static func + (lhs: NDArray, rhs: NDArray) -> NDArray {
        lhs.add(rhs)
    }

    public static func - (lhs: NDArray, rhs: NDArray) -> NDArray {
        lhs.subtract(rhs)
    }

    public static func * (lhs: NDArray, rhs: NDArray) -> NDArray {
        lhs.multiply(rhs)
    }

    public static func / (lhs: NDArray, rhs: NDArray) -> NDArray {
        lhs.divide(rhs)
    }

    public static prefix func - (array: NDArray) -> NDArray {
        array.negated()
    }

    public static func + (lhs: NDArray, rhs: Double) -> NDArray {
        lhs.add(rhs)
    }

    public static func + (lhs: Double, rhs: NDArray) -> NDArray {
        rhs.add(lhs)
    }

    public static func - (lhs: NDArray, rhs: Double) -> NDArray {
        lhs.subtract(rhs)
    }

    public static func - (lhs: Double, rhs: NDArray) -> NDArray {
        rhs.negated().add(lhs)
    }

    public static func * (lhs: NDArray, rhs: Double) -> NDArray {
        lhs.multiply(rhs)
    }

    public static func * (lhs: Double, rhs: NDArray) -> NDArray {
        rhs.multiply(lhs)
    }

    public static func / (lhs: NDArray, rhs: Double) -> NDArray {
        lhs.divide(rhs)
    }

    public static func / (lhs: Double, rhs: NDArray) -> NDArray {
        rhs.scalarDivide(lhs)
    }
}
