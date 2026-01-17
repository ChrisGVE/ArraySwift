//
//  NDArrayMath.swift
//  ArraySwift
//
//  Mathematical functions for NDArray using vDSP
//

import Foundation
import Accelerate

// MARK: - Mathematical Functions

extension NDArray {

    // MARK: - Basic Math

    /// Absolute value of each element.
    public func abs() -> NDArray {
        if isComplex, let imagPart = imag {
            var result = [Double](repeating: 0, count: size)
            real.withUnsafeBufferPointer { realBuf in
                imagPart.withUnsafeBufferPointer { imagBuf in
                    result.withUnsafeMutableBufferPointer { resBuf in
                        var split = DSPDoubleSplitComplex(
                            realp: UnsafeMutablePointer(mutating: realBuf.baseAddress!),
                            imagp: UnsafeMutablePointer(mutating: imagBuf.baseAddress!)
                        )
                        vDSP_zvabsD(&split, 1, resBuf.baseAddress!, 1, vDSP_Length(size))
                    }
                }
            }
            return NDArray(shape: shape, data: result)
        }

        var result = [Double](repeating: 0, count: size)
        vDSP_vabsD(real, 1, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    /// Square root of each element.
    /// For complex arrays, returns complex square root.
    /// For real arrays with negative values, returns NaN (use csqrt for complex result).
    public func sqrt() -> NDArray {
        if isComplex {
            return csqrt()
        }

        var result = real
        var count = Int32(size)
        vvsqrt(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Square of each element.
    public func square() -> NDArray {
        if isComplex, let imagPart = imag {
            // (a + bi)² = a² - b² + 2abi
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                let a = real[i]
                let b = imagPart[i]
                resultReal[i] = a * a - b * b
                resultImag[i] = 2 * a * b
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = [Double](repeating: 0, count: size)
        vDSP_vsqD(real, 1, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    /// Sign of each element (-1, 0, or 1 for real; z/|z| for complex).
    public func sign() -> NDArray {
        if isComplex, let imagPart = imag {
            // sign(z) = z / |z| (normalized to unit circle, 0 if z = 0)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                let r = hypot(real[i], imagPart[i])
                if r > 0 {
                    resultReal[i] = real[i] / r
                    resultImag[i] = imagPart[i] / r
                }
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            if real[i] > 0 { result[i] = 1 }
            else if real[i] < 0 { result[i] = -1 }
            else { result[i] = 0 }
        }
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Exponential and Logarithmic

    /// Exponential of each element.
    public func exp() -> NDArray {
        if isComplex, let imagPart = imag {
            // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                let expA = Darwin.exp(real[i])
                resultReal[i] = expA * Darwin.cos(imagPart[i])
                resultImag[i] = expA * Darwin.sin(imagPart[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvexp(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Natural logarithm of each element.
    public func log() -> NDArray {
        if isComplex, let imagPart = imag {
            // log(a + bi) = log(|z|) + i*arg(z)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                let r = hypot(real[i], imagPart[i])
                resultReal[i] = Darwin.log(r)
                resultImag[i] = atan2(imagPart[i], real[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvlog(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Base-2 logarithm of each element.
    public func log2() -> NDArray {
        var result = real
        var count = Int32(size)
        vvlog2(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Base-10 logarithm of each element.
    public func log10() -> NDArray {
        var result = real
        var count = Int32(size)
        vvlog10(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Natural logarithm of (1 + x) for each element.
    public func log1p() -> NDArray {
        var result = real
        var count = Int32(size)
        vvlog1p(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// exp(x) - 1 for each element.
    public func expm1() -> NDArray {
        var result = real
        var count = Int32(size)
        vvexpm1(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Trigonometric Functions

    /// Sine of each element.
    public func sin() -> NDArray {
        if isComplex, let imagPart = imag {
            // sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                resultReal[i] = Darwin.sin(real[i]) * Darwin.cosh(imagPart[i])
                resultImag[i] = Darwin.cos(real[i]) * Darwin.sinh(imagPart[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvsin(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Cosine of each element.
    public func cos() -> NDArray {
        if isComplex, let imagPart = imag {
            // cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                resultReal[i] = Darwin.cos(real[i]) * Darwin.cosh(imagPart[i])
                resultImag[i] = -Darwin.sin(real[i]) * Darwin.sinh(imagPart[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvcos(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Tangent of each element.
    public func tan() -> NDArray {
        if isComplex {
            // tan(z) = sin(z) / cos(z)
            let s = self.sin()
            let c = self.cos()
            return s.divide(c)
        }

        var result = real
        var count = Int32(size)
        vvtan(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Arcsine of each element.
    public func arcsin() -> NDArray {
        var result = real
        var count = Int32(size)
        vvasin(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Arccosine of each element.
    public func arccos() -> NDArray {
        var result = real
        var count = Int32(size)
        vvacos(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Arctangent of each element.
    public func arctan() -> NDArray {
        var result = real
        var count = Int32(size)
        vvatan(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Arctangent of y/x considering quadrant.
    public func arctan2(_ other: NDArray) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var count = Int32(size)
        vvatan2(&result, real, other.real, &count)
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Hyperbolic Functions

    /// Hyperbolic sine of each element.
    public func sinh() -> NDArray {
        if isComplex, let imagPart = imag {
            // sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                resultReal[i] = Darwin.sinh(real[i]) * Darwin.cos(imagPart[i])
                resultImag[i] = Darwin.cosh(real[i]) * Darwin.sin(imagPart[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvsinh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Hyperbolic cosine of each element.
    public func cosh() -> NDArray {
        if isComplex, let imagPart = imag {
            // cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
            var resultReal = [Double](repeating: 0, count: size)
            var resultImag = [Double](repeating: 0, count: size)

            for i in 0..<size {
                resultReal[i] = Darwin.cosh(real[i]) * Darwin.cos(imagPart[i])
                resultImag[i] = Darwin.sinh(real[i]) * Darwin.sin(imagPart[i])
            }

            return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
        }

        var result = real
        var count = Int32(size)
        vvcosh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Hyperbolic tangent of each element.
    public func tanh() -> NDArray {
        if isComplex {
            // tanh(z) = sinh(z) / cosh(z)
            let s = self.sinh()
            let c = self.cosh()
            return s.divide(c)
        }

        var result = real
        var count = Int32(size)
        vvtanh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Inverse hyperbolic sine of each element.
    public func arcsinh() -> NDArray {
        if isComplex {
            // arcsinh(z) = log(z + sqrt(z² + 1))
            let zSquared = self.square()
            let one = NDArray.full(shape, real: 1, imag: 0)
            let inner = zSquared.add(one)
            let sqrtInner = inner.csqrt()
            let sum = self.add(sqrtInner)
            return sum.clog()
        }

        var result = real
        var count = Int32(size)
        vvasinh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Inverse hyperbolic cosine of each element.
    public func arccosh() -> NDArray {
        if isComplex {
            // arccosh(z) = log(z + sqrt(z² - 1))
            let zSquared = self.square()
            let one = NDArray.full(shape, real: 1, imag: 0)
            let inner = zSquared.subtract(one)
            let sqrtInner = inner.csqrt()
            let sum = self.add(sqrtInner)
            return sum.clog()
        }

        var result = real
        var count = Int32(size)
        vvacosh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Inverse hyperbolic tangent of each element.
    public func arctanh() -> NDArray {
        if isComplex {
            // arctanh(z) = 0.5 * log((1 + z) / (1 - z))
            let one = NDArray.full(shape, real: 1, imag: 0)
            let onePlusZ = one.add(self)
            let oneMinusZ = one.subtract(self)
            let ratio = onePlusZ.divide(oneMinusZ)
            let logRatio = ratio.clog()
            return logRatio.multiply(0.5)
        }

        var result = real
        var count = Int32(size)
        vvatanh(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Rounding Functions

    /// Floor of each element.
    public func floor() -> NDArray {
        var result = real
        var count = Int32(size)
        vvfloor(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Ceiling of each element.
    public func ceil() -> NDArray {
        var result = real
        var count = Int32(size)
        vvceil(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Round each element to nearest integer.
    public func round() -> NDArray {
        var result = real
        var count = Int32(size)
        vvnint(&result, real, &count)
        return NDArray(shape: shape, data: result)
    }

    /// Truncate each element towards zero.
    public func trunc() -> NDArray {
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = Darwin.trunc(real[i])
        }
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Clipping

    /// Clip values to a range.
    public func clip(min: Double, max: Double) -> NDArray {
        var result = [Double](repeating: 0, count: size)
        var low = min
        var high = max
        vDSP_vclipD(real, 1, &low, &high, &result, 1, vDSP_Length(size))
        return NDArray(shape: shape, data: result)
    }

    // MARK: - Complex-Specific Functions

    /// Phase angle (argument) of complex elements.
    public func angle() -> NDArray {
        if isComplex, let imagPart = imag {
            var result = [Double](repeating: 0, count: size)
            real.withUnsafeBufferPointer { realBuf in
                imagPart.withUnsafeBufferPointer { imagBuf in
                    result.withUnsafeMutableBufferPointer { resBuf in
                        var split = DSPDoubleSplitComplex(
                            realp: UnsafeMutablePointer(mutating: realBuf.baseAddress!),
                            imagp: UnsafeMutablePointer(mutating: imagBuf.baseAddress!)
                        )
                        vDSP_zvphasD(&split, 1, resBuf.baseAddress!, 1, vDSP_Length(size))
                    }
                }
            }
            return NDArray(shape: shape, data: result)
        }

        // For real arrays, return atan2(0, x) = 0 or π
        var result = [Double](repeating: 0, count: size)
        for i in 0..<size {
            result[i] = real[i] >= 0 ? 0 : .pi
        }
        return NDArray(shape: shape, data: result)
    }

    /// Complex conjugate.
    public func conjugate() -> NDArray {
        guard isComplex, let imagPart = imag else { return self }

        var resultImag = [Double](repeating: 0, count: size)
        vDSP_vnegD(imagPart, 1, &resultImag, 1, vDSP_Length(size))

        return NDArray(shape: shape, dtype: .complex128, real: real, imag: resultImag)
    }

    /// Extract real part.
    public func realPart() -> NDArray {
        NDArray(shape: shape, data: real)
    }

    /// Extract imaginary part.
    public func imagPart() -> NDArray {
        let imagData = imag ?? [Double](repeating: 0, count: size)
        return NDArray(shape: shape, data: imagData)
    }

    // MARK: - Complex Square Root and Log

    /// Complex square root (handles negative numbers).
    public func csqrt() -> NDArray {
        var resultReal = [Double](repeating: 0, count: size)
        var resultImag = [Double](repeating: 0, count: size)

        let realPart = self.real
        let imagPart = self.imag ?? [Double](repeating: 0, count: size)

        for i in 0..<size {
            let r = hypot(realPart[i], imagPart[i])
            let theta = atan2(imagPart[i], realPart[i])
            let sqrtR = Darwin.sqrt(r)
            let halfTheta = theta / 2
            resultReal[i] = sqrtR * Darwin.cos(halfTheta)
            resultImag[i] = sqrtR * Darwin.sin(halfTheta)
        }

        return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
    }

    /// Complex logarithm.
    public func clog() -> NDArray {
        var resultReal = [Double](repeating: 0, count: size)
        var resultImag = [Double](repeating: 0, count: size)

        let realPart = self.real
        let imagPart = self.imag ?? [Double](repeating: 0, count: size)

        for i in 0..<size {
            let r = hypot(realPart[i], imagPart[i])
            resultReal[i] = Darwin.log(r)
            resultImag[i] = atan2(imagPart[i], realPart[i])
        }

        return NDArray.complexArray(shape: shape, real: resultReal, imag: resultImag)
    }
}

// MARK: - Free Functions (NumPy-style API)

/// Absolute value of array elements.
public func abs(_ array: NDArray) -> NDArray { array.abs() }

/// Square root of array elements.
public func sqrt(_ array: NDArray) -> NDArray { array.sqrt() }

/// Square of array elements.
public func square(_ array: NDArray) -> NDArray { array.square() }

/// Sign of array elements.
public func sign(_ array: NDArray) -> NDArray { array.sign() }

/// Exponential of array elements.
public func exp(_ array: NDArray) -> NDArray { array.exp() }

/// Natural logarithm of array elements.
public func log(_ array: NDArray) -> NDArray { array.log() }

/// Base-2 logarithm of array elements.
public func log2(_ array: NDArray) -> NDArray { array.log2() }

/// Base-10 logarithm of array elements.
public func log10(_ array: NDArray) -> NDArray { array.log10() }

/// log(1 + x) of array elements.
public func log1p(_ array: NDArray) -> NDArray { array.log1p() }

/// exp(x) - 1 of array elements.
public func expm1(_ array: NDArray) -> NDArray { array.expm1() }

/// Sine of array elements.
public func sin(_ array: NDArray) -> NDArray { array.sin() }

/// Cosine of array elements.
public func cos(_ array: NDArray) -> NDArray { array.cos() }

/// Tangent of array elements.
public func tan(_ array: NDArray) -> NDArray { array.tan() }

/// Arcsine of array elements.
public func arcsin(_ array: NDArray) -> NDArray { array.arcsin() }

/// Arccosine of array elements.
public func arccos(_ array: NDArray) -> NDArray { array.arccos() }

/// Arctangent of array elements.
public func arctan(_ array: NDArray) -> NDArray { array.arctan() }

/// Arctangent of y/x.
public func arctan2(_ y: NDArray, _ x: NDArray) -> NDArray { y.arctan2(x) }

/// Hyperbolic sine of array elements.
public func sinh(_ array: NDArray) -> NDArray { array.sinh() }

/// Hyperbolic cosine of array elements.
public func cosh(_ array: NDArray) -> NDArray { array.cosh() }

/// Hyperbolic tangent of array elements.
public func tanh(_ array: NDArray) -> NDArray { array.tanh() }

/// Inverse hyperbolic sine of array elements.
public func arcsinh(_ array: NDArray) -> NDArray { array.arcsinh() }

/// Inverse hyperbolic cosine of array elements.
public func arccosh(_ array: NDArray) -> NDArray { array.arccosh() }

/// Inverse hyperbolic tangent of array elements.
public func arctanh(_ array: NDArray) -> NDArray { array.arctanh() }

/// Floor of array elements.
public func floor(_ array: NDArray) -> NDArray { array.floor() }

/// Ceiling of array elements.
public func ceil(_ array: NDArray) -> NDArray { array.ceil() }

/// Round array elements.
public func round(_ array: NDArray) -> NDArray { array.round() }

/// Truncate array elements.
public func trunc(_ array: NDArray) -> NDArray { array.trunc() }

/// Clip array values to range.
public func clip(_ array: NDArray, min: Double, max: Double) -> NDArray { array.clip(min: min, max: max) }

/// Phase angle of complex array.
public func angle(_ array: NDArray) -> NDArray { array.angle() }

/// Complex conjugate.
public func conj(_ array: NDArray) -> NDArray { array.conjugate() }
