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
  /// Supports broadcasting and complex arrays.
  public func add(_ other: NDArray) -> NDArray {
    if isComplex || other.isComplex {
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexAdd(a, b)
    }

    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    vDSP_vaddD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
    return NDArray(shape: resultShape, data: result)
  }

  /// Subtract two arrays element-wise.
  /// Supports broadcasting and complex arrays.
  public func subtract(_ other: NDArray) -> NDArray {
    if isComplex || other.isComplex {
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexSub(a, b)
    }

    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    // vDSP_vsubD computes B - A
    vDSP_vsubD(b.real, 1, a.real, 1, &result, 1, vDSP_Length(a.size))
    return NDArray(shape: resultShape, data: result)
  }

  /// Multiply two arrays element-wise.
  /// Supports broadcasting and complex arrays.
  public func multiply(_ other: NDArray) -> NDArray {
    if isComplex || other.isComplex {
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexMul(a, b)
    }

    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    vDSP_vmulD(a.real, 1, b.real, 1, &result, 1, vDSP_Length(a.size))
    return NDArray(shape: resultShape, data: result)
  }

  /// Divide two arrays element-wise.
  /// Supports broadcasting and complex arrays.
  public func divide(_ other: NDArray) -> NDArray {
    if isComplex || other.isComplex {
      let (a, b, _) = broadcast(self.promoteToComplex(), other.promoteToComplex())
      return complexDiv(a, b)
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

  // MARK: - Dot Product

  /// Compute the dot product of two arrays.
  /// - For 1-D arrays: inner product of vectors
  /// - For 2-D arrays: matrix multiplication
  /// - For N-D arrays: sum product over last axis of a and second-to-last of b
  public func dot(_ other: NDArray) -> NDArray {
    // 1D x 1D: inner product (returns scalar as 0-D array)
    if ndim == 1 && other.ndim == 1 {
      precondition(size == other.size, "Vectors must have same length for dot product")
      var result: Double = 0
      vDSP_dotprD(real, 1, other.real, 1, &result, vDSP_Length(size))
      return NDArray(shape: [1], data: [result])
    }

    // 2D x 2D: matrix multiplication
    if ndim == 2 && other.ndim == 2 {
      return matmul(other)
    }

    // 2D x 1D: matrix-vector multiplication
    if ndim == 2 && other.ndim == 1 {
      precondition(shape[1] == other.size, "Inner dimensions must match")
      return matvec(other)
    }

    // 1D x 2D: vector-matrix multiplication
    if ndim == 1 && other.ndim == 2 {
      precondition(size == other.shape[0], "Inner dimensions must match")
      return vecmat(other)
    }

    // General N-D case
    return generalDot(other)
  }

  /// Matrix multiplication (2D x 2D).
  public func matmul(_ other: NDArray) -> NDArray {
    precondition(ndim == 2 && other.ndim == 2, "Both arrays must be 2D for matmul")
    precondition(
      shape[1] == other.shape[0], "Inner dimensions must match: \(shape[1]) != \(other.shape[0])")

    let m = shape[0]  // rows of A
    let k = shape[1]  // cols of A = rows of B
    let n = other.shape[1]  // cols of B

    var result = [Double](repeating: 0, count: m * n)

    // Use vDSP for matrix multiply: C = A * B
    // vDSP_mmulD computes C = A * B where:
    // A is m x k, B is k x n, C is m x n
    vDSP_mmulD(
      real, 1,
      other.real, 1,
      &result, 1,
      vDSP_Length(m),
      vDSP_Length(n),
      vDSP_Length(k)
    )

    return NDArray(shape: [m, n], data: result)
  }

  /// Matrix-vector multiplication (2D x 1D).
  private func matvec(_ vec: NDArray) -> NDArray {
    let m = shape[0]
    let n = shape[1]

    var result = [Double](repeating: 0, count: m)

    for i in 0..<m {
      var dot: Double = 0
      for j in 0..<n {
        dot += real[i * n + j] * vec.real[j]
      }
      result[i] = dot
    }

    return NDArray(shape: [m], data: result)
  }

  /// Vector-matrix multiplication (1D x 2D).
  private func vecmat(_ mat: NDArray) -> NDArray {
    let m = mat.shape[0]
    let n = mat.shape[1]

    var result = [Double](repeating: 0, count: n)

    for j in 0..<n {
      var dot: Double = 0
      for i in 0..<m {
        dot += real[i] * mat.real[i * n + j]
      }
      result[j] = dot
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

  /// Compute the inner product of two arrays.
  /// For 1-D arrays, this is the same as dot product.
  /// For N-D arrays, this sums over the last axes of both arrays.
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

  /// Compute the outer product of two vectors.
  /// Returns a matrix where result[i,j] = a[i] * b[j].
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
      precondition(false, "Cannot broadcast shapes \(a.shape) and \(b.shape)")
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
  var resultImag: [Double]? = array.isComplex
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
  /// For complex arrays, an element is non-zero if either real or imaginary part is non-zero.
  public func logicalAnd(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      let aTruthy = a.isNonZero(at: i)
      let bTruthy = b.isNonZero(at: i)
      result[i] = (aTruthy && bTruthy) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical OR.
  /// Treats non-zero as true, returns 1.0 for true, 0.0 for false.
  /// For complex arrays, an element is non-zero if either real or imaginary part is non-zero.
  public func logicalOr(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      let aTruthy = a.isNonZero(at: i)
      let bTruthy = b.isNonZero(at: i)
      result[i] = (aTruthy || bTruthy) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical XOR.
  /// Treats non-zero as true, returns 1.0 for true, 0.0 for false.
  /// For complex arrays, an element is non-zero if either real or imaginary part is non-zero.
  public func logicalXor(_ other: NDArray) -> NDArray {
    let (a, b, resultShape) = broadcast(self, other)
    var result = [Double](repeating: 0, count: a.size)
    for i in 0..<a.size {
      let aTruthy = a.isNonZero(at: i)
      let bTruthy = b.isNonZero(at: i)
      result[i] = (aTruthy != bTruthy) ? 1.0 : 0.0
    }
    return NDArray(shape: resultShape, data: result)
  }

  /// Element-wise logical NOT.
  /// Treats non-zero as true, returns 1.0 where false, 0.0 where true.
  /// For complex arrays, an element is non-zero if either real or imaginary part is non-zero.
  public func logicalNot() -> NDArray {
    var result = [Double](repeating: 0, count: size)
    for i in 0..<size {
      result[i] = isNonZero(at: i) ? 0.0 : 1.0
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
