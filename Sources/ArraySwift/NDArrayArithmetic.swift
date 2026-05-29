//
//  NDArrayArithmetic.swift
//  ArraySwift
//
//  Arithmetic operations for NDArray using vDSP
//

import Accelerate
import Foundation

// MARK: - Arithmetic Operations

extension NDArray {

  // MARK: - Negation

  /// Negate all elements.
  public func negated() -> NDArray {
    switch storage {
    case .int64(let d):
      return NDArray(shape: shape, int64Data: d.map { 0 &- $0 })
    case .bool(let d):
      // Negating bool treats false(0) as 0 and true(1) as -1 after int promotion
      return NDArray(shape: shape, int64Data: d.map { Int64(0) &- Int64($0) })
    case .complex128(let r, let im):
      var negR = [Double](repeating: 0, count: size)
      var negI = [Double](repeating: 0, count: size)
      vDSP_vnegD(r, 1, &negR, 1, vDSP_Length(size))
      vDSP_vnegD(im, 1, &negI, 1, vDSP_Length(size))
      return NDArray(shape: shape, storage: .complex128(real: negR, imag: negI))
    default:
      var result = [Double](repeating: 0, count: size)
      vDSP_vnegD(real, 1, &result, 1, vDSP_Length(size))
      return NDArray(shape: shape, data: result)
    }
  }

  // MARK: - Element-wise Binary Operations

  /// Add two arrays element-wise.
  ///
  /// Promotes operands to their common dtype before operating. Supports
  /// broadcasting, complex, int64, and bool arrays.
  public func add(_ other: NDArray) -> NDArray {
    let target = ArrayDType.promote(dtype, other.dtype)
    switch target {
    case .complex128:
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexAdd(a, b)
    case .int64:
      let (a, b, _) = broadcast(self.promoted(to: .int64), other.promoted(to: .int64))
      return NDArray.int64Add(a, b)
    default:
      let (a, b, resultShape) = broadcast(self.promoted(to: .float64), other.promoted(to: .float64))
      var result = [Double](repeating: 0, count: a.size)
      vDSP_vaddD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
      return NDArray(shape: resultShape, data: result)
    }
  }

  /// Subtract two arrays element-wise.
  ///
  /// Promotes operands to their common dtype before operating. Supports
  /// broadcasting, complex, int64, and bool arrays.
  public func subtract(_ other: NDArray) -> NDArray {
    let target = ArrayDType.promote(dtype, other.dtype)
    switch target {
    case .complex128:
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexSub(a, b)
    case .int64:
      let (a, b, _) = broadcast(self.promoted(to: .int64), other.promoted(to: .int64))
      return NDArray.int64Sub(a, b)
    default:
      let (a, b, resultShape) = broadcast(self.promoted(to: .float64), other.promoted(to: .float64))
      var result = [Double](repeating: 0, count: a.size)
      // vDSP_vsubD computes B - A
      vDSP_vsubD(b.real, 1, a.real, 1, &result, 1, vDSP_Length(a.size))
      return NDArray(shape: resultShape, data: result)
    }
  }

  /// Multiply two arrays element-wise.
  ///
  /// Promotes operands to their common dtype before operating. Supports
  /// broadcasting, complex, int64, and bool arrays.
  public func multiply(_ other: NDArray) -> NDArray {
    let target = ArrayDType.promote(dtype, other.dtype)
    switch target {
    case .complex128:
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexMul(a, b)
    case .int64:
      let (a, b, _) = broadcast(self.promoted(to: .int64), other.promoted(to: .int64))
      return NDArray.int64Mul(a, b)
    default:
      let (a, b, resultShape) = broadcast(self.promoted(to: .float64), other.promoted(to: .float64))
      var result = [Double](repeating: 0, count: a.size)
      vDSP_vmulD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
      return NDArray(shape: resultShape, data: result)
    }
  }

  /// Divide two arrays element-wise.
  ///
  /// Promotes operands to their common dtype before operating. Supports
  /// broadcasting, complex, int64, and bool arrays.
  public func divide(_ other: NDArray) -> NDArray {
    let target = ArrayDType.promote(dtype, other.dtype)
    switch target {
    case .complex128:
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexDiv(a, b)
    case .int64:
      let (a, b, _) = broadcast(self.promoted(to: .int64), other.promoted(to: .int64))
      return NDArray.int64Div(a, b)
    default:
      let (a, b, resultShape) = broadcast(self.promoted(to: .float64), other.promoted(to: .float64))
      var result = [Double](repeating: 0, count: a.size)
      // vDSP_vdivD computes B / A
      vDSP_vdivD(b.real, 1, a.real, 1, &result, 1, vDSP_Length(a.size))
      return NDArray(shape: resultShape, data: result)
    }
  }

  /// Raise every element to a scalar power. Equivalent to NumPy's `numpy.power`.
  ///
  /// Only supported for real arrays.
  ///
  /// - Parameter exponent: Scalar exponent applied to each element.
  /// - Returns: A real `NDArray` of powered values.
  public func power(_ exponent: Double) -> NDArray {
    var result = [Double](repeating: 0, count: size)
    for i in 0..<size {
      result[i] = pow(real[i], exponent)
    }
    return NDArray(shape: shape, data: result)
  }

  /// Element-wise power with broadcasting. Equivalent to NumPy's `numpy.power`.
  ///
  /// Only supported for real arrays.
  ///
  /// - Parameter other: Exponent array (broadcast-compatible).
  /// - Returns: A real `NDArray` where each element is `self[i] ** other[i]`.
  public func power(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      result[i] = pow(a.real[i], b.real[i])
    }
    return NDArray(shape: resultShape, data: result)
  }

  // MARK: - Scalar Operations

  /// Add a scalar to every element. Uses vDSP for vectorised execution.
  /// - Parameter scalar: Value to add.
  /// - Returns: A new `NDArray` with the scalar added to each element.
  public func add(_ scalar: Double) -> NDArray {
    var result = [Double](repeating: 0, count: size)
    var s = scalar
    vDSP_vsaddD(real, 1, &s, &result, 1, vDSP_Length(size))
    return NDArray(shape: shape, data: result)
  }

  /// Subtract a scalar from every element.
  /// - Parameter scalar: Value to subtract.
  /// - Returns: A new `NDArray` with the scalar subtracted from each element.
  public func subtract(_ scalar: Double) -> NDArray {
    add(-scalar)
  }

  /// Multiply every element by a scalar. Uses vDSP for vectorised execution.
  /// - Parameter scalar: Scaling factor.
  /// - Returns: A new `NDArray` with each element multiplied by the scalar.
  public func multiply(_ scalar: Double) -> NDArray {
    var result = [Double](repeating: 0, count: size)
    var s = scalar
    vDSP_vsmulD(real, 1, &s, &result, 1, vDSP_Length(size))
    return NDArray(shape: shape, data: result)
  }

  /// Divide every element by a scalar. Uses vDSP for vectorised execution.
  /// - Parameter scalar: Divisor.
  /// - Returns: A new `NDArray` with each element divided by the scalar.
  public func divide(_ scalar: Double) -> NDArray {
    var result = [Double](repeating: 0, count: size)
    var s = scalar
    vDSP_vsdivD(real, 1, &s, &result, 1, vDSP_Length(size))
    return NDArray(shape: shape, data: result)
  }

  /// Compute `scalar / x` for every element `x`. Uses vDSP for vectorised execution.
  /// - Parameter scalar: Dividend.
  /// - Returns: A new `NDArray` where each element is `scalar / element`.
  public func scalarDivide(_ scalar: Double) -> NDArray {
    var result = [Double](repeating: 0, count: size)
    var s = scalar
    vDSP_svdivD(&s, real, 1, &result, 1, vDSP_Length(size))
    return NDArray(shape: shape, data: result)
  }

  // MARK: - Modulo Operations

  /// Element-wise truncating remainder (C-style `%`). Equivalent to NumPy's `numpy.fmod`.
  ///
  /// The result has the same sign as the dividend. For Python-style (floor) modulo use
  /// ``floorMod(_:)``.
  ///
  /// - Parameter other: Divisor array (broadcast-compatible).
  /// - Returns: A real `NDArray` of remainders.
  public func mod(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      result[i] = a.real[i].truncatingRemainder(dividingBy: b.real[i])
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise floor modulo (Python-style `%`). Equivalent to NumPy's `numpy.mod`.
  ///
  /// The result has the same sign as the divisor, matching Python's `%` semantics.
  ///
  /// - Parameter other: Divisor array (broadcast-compatible).
  /// - Returns: A real `NDArray` of floor remainders.
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

  // MARK: - Dot Product

  /// Compute the dot product of two arrays.
  /// - For 1-D arrays: inner product of vectors (complex-aware)
  /// - For 2-D arrays: matrix multiplication (complex-aware)
  /// - For N-D arrays: sum product over last axis of a and second-to-last of b
  public func dot(_ other: NDArray) -> NDArray {
    let useComplex = isComplex || other.isComplex
    let a = useComplex ? promoteToComplex() : self
    let b = useComplex ? other.promoteToComplex() : other

    // 1D x 1D: inner product
    if a.ndim == 1 && b.ndim == 1 {
      precondition(a.size == b.size, "Vectors must have same length for dot product")
      if useComplex {
        return complexDot1D(a, b)
      }
      var result: Double = 0
      vDSP_dotprD(a.real, 1, b.real, 1, &result, vDSP_Length(a.size))
      return NDArray(shape: [1], data: [result])
    }

    // 2D x 2D: matrix multiplication
    if a.ndim == 2 && b.ndim == 2 {
      return a.matmul(b)
    }

    // 2D x 1D: matrix-vector multiplication
    if a.ndim == 2 && b.ndim == 1 {
      precondition(a.shape[1] == b.size, "Inner dimensions must match")
      return a.matvec(b)
    }

    // 1D x 2D: vector-matrix multiplication
    if a.ndim == 1 && b.ndim == 2 {
      precondition(a.size == b.shape[0], "Inner dimensions must match")
      return a.vecmat(b)
    }

    // General N-D case
    return a.generalDot(b)
  }

  /// Matrix multiplication (2D x 2D). Supports complex arrays.
  public func matmul(_ other: NDArray) -> NDArray {
    precondition(ndim == 2 && other.ndim == 2, "Both arrays must be 2D for matmul")
    precondition(
      shape[1] == other.shape[0], "Inner dimensions must match: \(shape[1]) != \(other.shape[0])")

    let m = shape[0]
    let k = shape[1]
    let n = other.shape[1]

    if isComplex || other.isComplex {
      let a = promoteToComplex()
      let b = other.promoteToComplex()
      return complexMatmul(a, b, m: m, k: k, n: n)
    }

    var result = [Double](repeating: 0, count: m * n)
    vDSP_mmulD(
      real, 1, other.real, 1, &result, 1,
      vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    return NDArray(shape: [m, n], data: result)
  }

  /// Matrix-vector multiplication (2D x 1D). Supports complex arrays.
  private func matvec(_ vec: NDArray) -> NDArray {
    let m = shape[0]
    let n = shape[1]

    if isComplex || vec.isComplex {
      let matImag = imag ?? [Double](repeating: 0, count: size)
      let vecImag = vec.imag ?? [Double](repeating: 0, count: vec.size)
      var resultReal = [Double](repeating: 0, count: m)
      var resultImag = [Double](repeating: 0, count: m)
      for i in 0..<m {
        for j in 0..<n {
          let ar = real[i * n + j]
          let ai = matImag[i * n + j]
          let br = vec.real[j]
          let bi = vecImag[j]
          resultReal[i] += ar * br - ai * bi
          resultImag[i] += ar * bi + ai * br
        }
      }
      return NDArray.complexArray(shape: [m], real: resultReal, imag: resultImag)
    }

    var result = [Double](repeating: 0, count: m)
    for i in 0..<m {
      for j in 0..<n {
        result[i] += real[i * n + j] * vec.real[j]
      }
    }
    return NDArray(shape: [m], data: result)
  }

  /// Vector-matrix multiplication (1D x 2D). Supports complex arrays.
  private func vecmat(_ mat: NDArray) -> NDArray {
    let m = mat.shape[0]
    let n = mat.shape[1]

    if isComplex || mat.isComplex {
      let vecImag = imag ?? [Double](repeating: 0, count: size)
      let matImag = mat.imag ?? [Double](repeating: 0, count: mat.size)
      var resultReal = [Double](repeating: 0, count: n)
      var resultImag = [Double](repeating: 0, count: n)
      for j in 0..<n {
        for i in 0..<m {
          let ar = real[i]
          let ai = vecImag[i]
          let br = mat.real[i * n + j]
          let bi = matImag[i * n + j]
          resultReal[j] += ar * br - ai * bi
          resultImag[j] += ar * bi + ai * br
        }
      }
      return NDArray.complexArray(shape: [n], real: resultReal, imag: resultImag)
    }

    var result = [Double](repeating: 0, count: n)
    for j in 0..<n {
      for i in 0..<m {
        result[j] += real[i] * mat.real[i * n + j]
      }
    }
    return NDArray(shape: [n], data: result)
  }

  /// General N-D dot product: sum over last axis of a and second-to-last of b.
  private func generalDot(_ other: NDArray) -> NDArray {
    // Sum over last axis of self and second-to-last axis of other
    // For now, implement a simplified version for common cases
    let aLastAxis = ndim - 1
    let bSecondLastAxis = other.ndim >= 2 ? other.ndim - 2 : 0

    precondition(
      shape[aLastAxis] == other.shape[bSecondLastAxis],
      "Contracting dimensions must match")

    // Build result shape: a.shape[:-1] + b.shape[:-2] + b.shape[-1:]
    var resultShape: [Int] = []
    for i in 0..<(ndim - 1) {
      resultShape.append(shape[i])
    }
    if other.ndim >= 2 {
      for i in 0..<(other.ndim - 2) {
        resultShape.append(other.shape[i])
      }
      resultShape.append(other.shape[other.ndim - 1])
    }
    if resultShape.isEmpty { resultShape = [1] }

    let k = shape[aLastAxis]  // contracting dimension

    // Compute the product
    let aOuterSize = shape.dropLast().reduce(1, *)
    let bOuterSize =
      other.ndim >= 2 ? other.shape.dropLast(2).reduce(1, *) * other.shape[other.ndim - 1] : 1

    var resultData = [Double](repeating: 0, count: resultShape.reduce(1, *))

    // Simple nested loop implementation
    let aStride = k
    let bColStride = other.ndim >= 2 ? other.shape[other.ndim - 1] : 1
    let bRowStride = other.ndim >= 2 ? k * bColStride : k

    for i in 0..<aOuterSize {
      for j in 0..<bOuterSize {
        var sum: Double = 0
        for m in 0..<k {
          let aIdx = i * aStride + m
          let bIdx: Int
          if other.ndim == 1 {
            bIdx = m
          } else {
            let bOuter = j / bColStride
            let bInner = j % bColStride
            bIdx = bOuter * bRowStride + m * bColStride + bInner
          }
          if aIdx < real.count && bIdx < other.real.count {
            sum += real[aIdx] * other.real[bIdx]
          }
        }
        let resultIdx = i * bOuterSize + j
        if resultIdx < resultData.count {
          resultData[resultIdx] = sum
        }
      }
    }

    return NDArray(shape: resultShape, data: resultData)
  }

  // MARK: - Cross Product

  /// Compute the cross product of two 3D vectors.
  /// Returns a vector perpendicular to both inputs.
  public func cross(_ other: NDArray) -> NDArray {
    precondition(size == 3 && other.size == 3, "Cross product requires 3-element vectors")

    let a = real
    let b = other.real

    // a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    let result = [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]

    return NDArray(shape: [3], data: result)
  }

  // MARK: - Inner Product

  /// Compute the inner product of two arrays. Equivalent to NumPy's `numpy.inner`.
  ///
  /// - For 1-D inputs: same as ``dot(_:)``.
  /// - For N-D inputs: contracts the last axis of both arrays, producing a result
  ///   with shape `self.shape[:-1] + other.shape[:-1]`.
  ///
  /// - Parameter other: Second operand. Last dimensions must match.
  /// - Returns: A new `NDArray` inner product.
  public func inner(_ other: NDArray) -> NDArray {
    if ndim == 1 && other.ndim == 1 {
      return dot(other)
    }

    // Sum over last axis of both arrays
    precondition(
      shape[ndim - 1] == other.shape[other.ndim - 1],
      "Last dimensions must match for inner product")

    // Build result shape
    var resultShape: [Int] = []
    for i in 0..<(ndim - 1) {
      resultShape.append(shape[i])
    }
    for i in 0..<(other.ndim - 1) {
      resultShape.append(other.shape[i])
    }
    if resultShape.isEmpty { resultShape = [1] }

    let k = shape[ndim - 1]  // contracting dimension
    let aOuterSize = shape.dropLast().reduce(1, *)
    let bOuterSize = other.shape.dropLast().reduce(1, *)

    var resultData = [Double](repeating: 0, count: resultShape.reduce(1, *))

    for i in 0..<aOuterSize {
      for j in 0..<bOuterSize {
        var sum: Double = 0
        for m in 0..<k {
          sum += real[i * k + m] * other.real[j * k + m]
        }
        resultData[i * bOuterSize + j] = sum
      }
    }

    return NDArray(shape: resultShape, data: resultData)
  }

  // MARK: - Outer Product

  /// Compute the outer product of two vectors. Equivalent to NumPy's `numpy.outer`.
  ///
  /// Both inputs are first flattened to 1-D. Returns a 2-D matrix `C` where
  /// `C[i, j] = self[i] * other[j]`.
  ///
  /// - Parameter other: Second vector.
  /// - Returns: A 2-D `NDArray` of shape `[self.size, other.size]`.
  public func outer(_ other: NDArray) -> NDArray {
    let flat = flatten()
    let otherFlat = other.flatten()

    var result = [Double](repeating: 0, count: flat.size * otherFlat.size)

    for i in 0..<flat.size {
      for j in 0..<otherFlat.size {
        result[i * otherFlat.size + j] = flat.real[i] * otherFlat.real[j]
      }
    }

    return NDArray(shape: [flat.size, otherFlat.size], data: result)
  }
}

// MARK: - Complex Dot / Matmul Helpers

/// 1-D complex inner product: sum(a_r*b_r - a_i*b_i) + i*sum(a_r*b_i + a_i*b_r).
private func complexDot1D(_ a: NDArray, _ b: NDArray) -> NDArray {
  let n = a.size
  let aImag = a.imag!
  let bImag = b.imag!
  var sumR = 0.0
  var sumI = 0.0
  for idx in 0..<n {
    sumR += a.real[idx] * b.real[idx] - aImag[idx] * bImag[idx]
    sumI += a.real[idx] * bImag[idx] + aImag[idx] * b.real[idx]
  }
  return NDArray.complexArray(shape: [1], real: [sumR], imag: [sumI])
}

/// 2-D complex matrix multiply using vDSP_zmmulD.
/// A is m×k, B is k×n, result C is m×n.
private func complexMatmul(_ a: NDArray, _ b: NDArray, m: Int, k: Int, n: Int) -> NDArray {
  var resultReal = [Double](repeating: 0, count: m * n)
  var resultImag = [Double](repeating: 0, count: m * n)

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
              vDSP_zmmulD(
                &splitA, 1, &splitB, 1, &splitC, 1,
                vDSP_Length(m), vDSP_Length(n), vDSP_Length(k)
              )
            }
          }
        }
      }
    }
  }

  return NDArray.complexArray(shape: [m, n], real: resultReal, imag: resultImag)
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
internal func broadcast(_ a: NDArray, _ b: NDArray) -> (NDArray, NDArray, [Int]) {
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
      precondition(false, "Cannot broadcast shapes \(a.shape) and \(b.shape)")
    }
  }

  let broadcastA = broadcastTo(a, shape: resultShape)
  let broadcastB = broadcastTo(b, shape: resultShape)

  return (broadcastA, broadcastB, resultShape)
}

/// Broadcast an array to a new shape.
internal func broadcastTo(_ array: NDArray, shape: [Int]) -> NDArray {
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
  var resultImag: [Double]? =
    array.isComplex
    ? [Double](repeating: 0, count: resultSize) : nil

  for i in 0..<resultSize {
    var sourceIdx = 0
    var remaining = i
    for d in 0..<shape.count {
      let coord = remaining / resultStrides[d]
      remaining = remaining % resultStrides[d]
      sourceIdx += (coord % paddedShape[d]) * sourceStrides[d]
    }
    result[i] = array.real[sourceIdx]
    if array.isComplex, let imagPart = array.imag {
      resultImag![i] = imagPart[sourceIdx]
    }
  }

  if array.isComplex {
    return NDArray(shape: shape, dtype: .complex128, real: result, imag: resultImag)
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
    if a.isComplex || b.isComplex {
      let aImag = a.imag ?? [Double](repeating: 0, count: a.size)
      let bImag = b.imag ?? [Double](repeating: 0, count: b.size)
      for i in 0..<a.size {
        result[i] = (a.real[i] == b.real[i] && aImag[i] == bImag[i]) ? 1.0 : 0.0
      }
    } else {
      for i in 0..<a.size {
        result[i] = a.real[i] == b.real[i] ? 1.0 : 0.0
      }
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise inequality comparison.
  /// Returns NDArray with 1.0 where not equal, 0.0 where equal.
  /// For complex arrays, checks both real and imaginary parts.
  public func notEqual(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    if a.isComplex || b.isComplex {
      let aImag = a.imag ?? [Double](repeating: 0, count: a.size)
      let bImag = b.imag ?? [Double](repeating: 0, count: b.size)
      for i in 0..<a.size {
        result[i] = (a.real[i] != b.real[i] || aImag[i] != bImag[i]) ? 1.0 : 0.0
      }
    } else {
      for i in 0..<a.size {
        result[i] = a.real[i] != b.real[i] ? 1.0 : 0.0
      }
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

  // MARK: - Where (conditional selection)

  /// Select elements from one of two arrays based on condition.
  /// Where condition is non-zero, return x, else return y.
  /// For complex conditions, an element is non-zero if either part is non-zero.
  /// Preserves complex dtype if either x or y is complex.
  public static func `where`(_ condition: NDArray, _ x: NDArray, _ y: NDArray) -> NDArray {
    let (cond, xArr, _) = broadcast(condition, x)
    let (condFinal, yArr, resultShape) = broadcast(cond, y)
    let (xFinal, _, _) = broadcast(xArr, yArr)

    let outputComplex = xFinal.isComplex || yArr.isComplex
    var resultReal = [Double](repeating: 0, count: condFinal.size)
    var resultImag: [Double]? = outputComplex ? [Double](repeating: 0, count: condFinal.size) : nil

    for i in 0..<condFinal.size {
      let truthy = condFinal.isNonZero(at: i)
      resultReal[i] = truthy ? xFinal.real[i] : yArr.real[i]
      if outputComplex {
        let xImag = xFinal.imag?[i] ?? 0
        let yImag = yArr.imag?[i] ?? 0
        resultImag![i] = truthy ? xImag : yImag
      }
    }

    if outputComplex {
      return NDArray(shape: resultShape, dtype: .complex128, real: resultReal, imag: resultImag)
    }
    return NDArray(shape: resultShape, data: resultReal)
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

// MARK: - Free Functions for Linear Algebra

/// Dot product of two arrays.
public func dot(_ a: NDArray, _ b: NDArray) -> NDArray { a.dot(b) }

/// Matrix multiplication.
public func matmul(_ a: NDArray, _ b: NDArray) -> NDArray { a.matmul(b) }

/// Cross product of two 3D vectors.
public func cross(_ a: NDArray, _ b: NDArray) -> NDArray { a.cross(b) }

/// Inner product of two arrays.
public func inner(_ a: NDArray, _ b: NDArray) -> NDArray { a.inner(b) }

/// Outer product of two arrays.
public func outer(_ a: NDArray, _ b: NDArray) -> NDArray { a.outer(b) }

// MARK: - Operators

extension NDArray {
  /// Element-wise addition with broadcasting. Equivalent to NumPy's `+` operator.
  public static func + (lhs: NDArray, rhs: NDArray) -> NDArray {
    lhs.add(rhs)
  }

  /// Element-wise subtraction with broadcasting. Equivalent to NumPy's `-` operator.
  public static func - (lhs: NDArray, rhs: NDArray) -> NDArray {
    lhs.subtract(rhs)
  }

  /// Element-wise multiplication with broadcasting. Equivalent to NumPy's `*` operator.
  public static func * (lhs: NDArray, rhs: NDArray) -> NDArray {
    lhs.multiply(rhs)
  }

  /// Element-wise division with broadcasting. Equivalent to NumPy's `/` operator.
  public static func / (lhs: NDArray, rhs: NDArray) -> NDArray {
    lhs.divide(rhs)
  }

  /// Unary negation of all elements.
  public static prefix func - (array: NDArray) -> NDArray {
    array.negated()
  }

  /// Add a scalar to every element.
  public static func + (lhs: NDArray, rhs: Double) -> NDArray {
    lhs.add(rhs)
  }

  /// Add a scalar to every element (scalar on the left).
  public static func + (lhs: Double, rhs: NDArray) -> NDArray {
    rhs.add(lhs)
  }

  /// Subtract a scalar from every element.
  public static func - (lhs: NDArray, rhs: Double) -> NDArray {
    lhs.subtract(rhs)
  }

  /// Subtract every element from a scalar (scalar on the left).
  public static func - (lhs: Double, rhs: NDArray) -> NDArray {
    rhs.negated().add(lhs)
  }

  /// Multiply every element by a scalar.
  public static func * (lhs: NDArray, rhs: Double) -> NDArray {
    lhs.multiply(rhs)
  }

  /// Multiply every element by a scalar (scalar on the left).
  public static func * (lhs: Double, rhs: NDArray) -> NDArray {
    rhs.multiply(lhs)
  }

  /// Divide every element by a scalar.
  public static func / (lhs: NDArray, rhs: Double) -> NDArray {
    lhs.divide(rhs)
  }

  /// Divide a scalar by every element (scalar on the left).
  public static func / (lhs: Double, rhs: NDArray) -> NDArray {
    rhs.scalarDivide(lhs)
  }

  // MARK: - Compound Assignment Operators

  /// Add `rhs` to `lhs` in place, with broadcasting.
  public static func += (lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs.add(rhs)
  }

  /// Subtract `rhs` from `lhs` in place, with broadcasting.
  public static func -= (lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs.subtract(rhs)
  }

  /// Multiply `lhs` by `rhs` in place, with broadcasting.
  public static func *= (lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs.multiply(rhs)
  }

  /// Divide `lhs` by `rhs` in place, with broadcasting.
  public static func /= (lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs.divide(rhs)
  }

  /// Add scalar `rhs` to every element of `lhs` in place.
  public static func += (lhs: inout NDArray, rhs: Double) {
    lhs = lhs.add(rhs)
  }

  /// Subtract scalar `rhs` from every element of `lhs` in place.
  public static func -= (lhs: inout NDArray, rhs: Double) {
    lhs = lhs.subtract(rhs)
  }

  /// Multiply every element of `lhs` by scalar `rhs` in place.
  public static func *= (lhs: inout NDArray, rhs: Double) {
    lhs = lhs.multiply(rhs)
  }

  /// Divide every element of `lhs` by scalar `rhs` in place.
  public static func /= (lhs: inout NDArray, rhs: Double) {
    lhs = lhs.divide(rhs)
  }
}
