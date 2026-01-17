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

// MARK: - Comparison Operations

extension NDArray {

    /// Element-wise equality comparison.
    /// Returns NDArray with 1.0 where equal, 0.0 where not equal.
    public func equal(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] == b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise inequality comparison.
    /// Returns NDArray with 1.0 where not equal, 0.0 where equal.
    public func notEqual(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] != b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise less than comparison.
    /// Returns NDArray with 1.0 where a < b, 0.0 otherwise.
    public func less(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] < b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise less than or equal comparison.
    /// Returns NDArray with 1.0 where a <= b, 0.0 otherwise.
    public func lessEqual(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] <= b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise greater than comparison.
    /// Returns NDArray with 1.0 where a > b, 0.0 otherwise.
    public func greater(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] > b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise greater than or equal comparison.
    /// Returns NDArray with 1.0 where a >= b, 0.0 otherwise.
    public func greaterEqual(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = a.real[i] >= b.real[i] ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise equality with scalar.
    public func equal(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] == scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise inequality with scalar.
    public func notEqual(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] != scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise less than with scalar.
    public func less(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] < scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise less than or equal with scalar.
    public func lessEqual(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] <= scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise greater than with scalar.
    public func greater(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] > scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise greater than or equal with scalar.
    public func greaterEqual(_ scalar: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] >= scalar ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Special Value Checks

    /// Element-wise check for NaN values.
    /// Returns NDArray with 1.0 where NaN, 0.0 otherwise.
    public func isnan() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i].isNaN ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise check for infinite values.
    /// Returns NDArray with 1.0 where infinite, 0.0 otherwise.
    public func isinf() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i].isInfinite ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise check for finite values.
    /// Returns NDArray with 1.0 where finite, 0.0 otherwise.
    public func isfinite() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i].isFinite ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise check for positive infinity.
    /// Returns NDArray with 1.0 where +inf, 0.0 otherwise.
    public func isposinf() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = (real[i].isInfinite && real[i] > 0) ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    /// Element-wise check for negative infinity.
    /// Returns NDArray with 1.0 where -inf, 0.0 otherwise.
    public func isneginf() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = (real[i].isInfinite && real[i] < 0) ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Logical Operations

    /// Element-wise logical AND.
    /// Treats non-zero as true, returns 1.0 for true, 0.0 for false.
    public func logicalAnd(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = (a.real[i] != 0 && b.real[i] != 0) ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise logical OR.
    /// Treats non-zero as true, returns 1.0 for true, 0.0 for false.
    public func logicalOr(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            result[i] = (a.real[i] != 0 || b.real[i] != 0) ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise logical XOR.
    /// Treats non-zero as true, returns 1.0 for true, 0.0 for false.
    public func logicalXor(_ other: NDArray) -> NDArray {
        let (a, b, resultShape) = broadcast(self, other)
        var result = [Double](repeating: 0, count: a.size)
        for i in 0..<a.size {
            let aTrue = a.real[i] != 0
            let bTrue = b.real[i] != 0
            result[i] = (aTrue != bTrue) ? 1.0 : 0.0
        }
        return NDArray(shape: resultShape, data: result)
    }

    /// Element-wise logical NOT.
    /// Treats non-zero as true, returns 1.0 where false, 0.0 where true.
    public func logicalNot() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] == 0 ? 1.0 : 0.0
        }
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Where (conditional selection)

    /// Select elements from one of two arrays based on condition.
    /// Where condition is non-zero, return x, else return y.
    public static func `where`(_ condition: NDArray, _ x: NDArray, _ y: NDArray) -> NDArray {
        let (cond, xArr, _) = broadcast(condition, x)
        let (condFinal, yArr, resultShape) = broadcast(cond, y)
        let (xFinal, _, _) = broadcast(xArr, yArr)

        var result = [Double](repeating: 0, count: condFinal.size)
        for i in 0..<condFinal.size {
            result[i] = condFinal.real[i] != 0 ? xFinal.real[i] : yArr.real[i]
        }
        return NDArray(shape: resultShape, data: result)
    }
}

// MARK: - Free Functions for Comparisons

/// Element-wise equality.
public func equal(_ a: NDArray, _ b: NDArray) -> NDArray { a.equal(b) }

/// Element-wise inequality.
public func notEqual(_ a: NDArray, _ b: NDArray) -> NDArray { a.notEqual(b) }

/// Element-wise less than.
public func less(_ a: NDArray, _ b: NDArray) -> NDArray { a.less(b) }

/// Element-wise less than or equal.
public func lessEqual(_ a: NDArray, _ b: NDArray) -> NDArray { a.lessEqual(b) }

/// Element-wise greater than.
public func greater(_ a: NDArray, _ b: NDArray) -> NDArray { a.greater(b) }

/// Element-wise greater than or equal.
public func greaterEqual(_ a: NDArray, _ b: NDArray) -> NDArray { a.greaterEqual(b) }

/// Element-wise NaN check.
public func isnan(_ array: NDArray) -> NDArray { array.isnan() }

/// Element-wise infinity check.
public func isinf(_ array: NDArray) -> NDArray { array.isinf() }

/// Element-wise finite check.
public func isfinite(_ array: NDArray) -> NDArray { array.isfinite() }

/// Element-wise logical AND.
public func logicalAnd(_ a: NDArray, _ b: NDArray) -> NDArray { a.logicalAnd(b) }

/// Element-wise logical OR.
public func logicalOr(_ a: NDArray, _ b: NDArray) -> NDArray { a.logicalOr(b) }

/// Element-wise logical XOR.
public func logicalXor(_ a: NDArray, _ b: NDArray) -> NDArray { a.logicalXor(b) }

/// Element-wise logical NOT.
public func logicalNot(_ array: NDArray) -> NDArray { array.logicalNot() }

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

    // MARK: - Compound Assignment Operators

    public static func += (lhs: inout NDArray, rhs: NDArray) {
        lhs = lhs.add(rhs)
    }

    public static func -= (lhs: inout NDArray, rhs: NDArray) {
        lhs = lhs.subtract(rhs)
    }

    public static func *= (lhs: inout NDArray, rhs: NDArray) {
        lhs = lhs.multiply(rhs)
    }

    public static func /= (lhs: inout NDArray, rhs: NDArray) {
        lhs = lhs.divide(rhs)
    }

    public static func += (lhs: inout NDArray, rhs: Double) {
        lhs = lhs.add(rhs)
    }

    public static func -= (lhs: inout NDArray, rhs: Double) {
        lhs = lhs.subtract(rhs)
    }

    public static func *= (lhs: inout NDArray, rhs: Double) {
        lhs = lhs.multiply(rhs)
    }

    public static func /= (lhs: inout NDArray, rhs: Double) {
        lhs = lhs.divide(rhs)
    }
}
