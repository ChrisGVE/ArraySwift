//
//  NDArrayReduction.swift
//  ArraySwift
//
//  Reduction operations for NDArray using vDSP
//

import Accelerate
import Foundation

// MARK: - Reduction Operations

extension NDArray {

  // MARK: - Sum and Product

  /// Sum of all elements (real part only for real arrays).
  public func sum() -> Double {
    var result: Double = 0
    vDSP_sveD(real, 1, &result, vDSP_Length(size))
    return result
  }

  /// Sum of all elements, returning complex for complex arrays.
  public func complexSum() -> (real: Double, imag: Double) {
    var realSum: Double = 0
    vDSP_sveD(real, 1, &realSum, vDSP_Length(size))

    if isComplex, let imagPart = imag {
      var imagSum: Double = 0
      vDSP_sveD(imagPart, 1, &imagSum, vDSP_Length(size))
      return (realSum, imagSum)
    }
    return (realSum, 0)
  }

  /// Sum along an axis.
  public func sum(axis: Int) -> NDArray {
    if isComplex {
      return reduceAlongAxisComplex(axis: axis) { base, axisStride, count in
        var realSum = 0.0
        var imagSum = 0.0
        let imagPart = imag!
        for i in 0..<count {
          realSum += real[base + i * axisStride]
          imagSum += imagPart[base + i * axisStride]
        }
        return (realSum, imagSum)
      }
    }
    return reduceAlongAxis(axis: axis) { base, axisStride, count in
      var result = 0.0
      for i in 0..<count { result += real[base + i * axisStride] }
      return result
    }
  }

  /// Product of all elements (real part only for real arrays).
  public func prod() -> Double {
    guard size > 0 else { return 1 }
    var result = real[0]
    for i in 1..<size {
      result *= real[i]
    }
    return result
  }

  /// Product of all elements, returning complex for complex arrays.
  public func complexProd() -> (real: Double, imag: Double) {
    guard size > 0 else { return (1, 0) }

    if isComplex, let imagPart = imag {
      var resultReal = real[0]
      var resultImag = imagPart[0]
      for i in 1..<size {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let a = resultReal
        let b = resultImag
        let c = real[i]
        let d = imagPart[i]
        resultReal = a * c - b * d
        resultImag = a * d + b * c
      }
      return (resultReal, resultImag)
    }
    return (prod(), 0)
  }

  /// Product along an axis.
  public func prod(axis: Int) -> NDArray {
    if isComplex {
      return reduceAlongAxisComplex(axis: axis) { base, axisStride, count in
        guard count > 0 else { return (1, 0) }
        let imagPart = imag!
        var resultReal = real[base]
        var resultImag = imagPart[base]
        for i in 1..<count {
          let idx = base + i * axisStride
          let a = resultReal
          let b = resultImag
          let c = real[idx]
          let d = imagPart[idx]
          resultReal = a * c - b * d
          resultImag = a * d + b * c
        }
        return (resultReal, resultImag)
      }
    }
    return reduceAlongAxis(axis: axis) { base, axisStride, count in
      var result = 1.0
      for i in 0..<count { result *= real[base + i * axisStride] }
      return result
    }
  }

  // MARK: - Mean and Variance

  /// Mean of all elements (real part only for real arrays).
  public func mean() -> Double {
    var result: Double = 0
    vDSP_meanvD(real, 1, &result, vDSP_Length(size))
    return result
  }

  /// Mean of all elements, returning complex for complex arrays.
  public func complexMean() -> (real: Double, imag: Double) {
    let s = complexSum()
    return (s.real / Double(size), s.imag / Double(size))
  }

  /// Mean along an axis.
  public func mean(axis: Int) -> NDArray {
    if isComplex {
      return reduceAlongAxisComplex(axis: axis) { base, axisStride, count in
        var realSum = 0.0
        var imagSum = 0.0
        let imagPart = imag!
        for i in 0..<count {
          realSum += real[base + i * axisStride]
          imagSum += imagPart[base + i * axisStride]
        }
        let n = Double(count)
        return (realSum / n, imagSum / n)
      }
    }
    return reduceAlongAxis(axis: axis) { base, axisStride, count in
      var result = 0.0
      for i in 0..<count { result += real[base + i * axisStride] }
      return result / Double(count)
    }
  }

  /// Variance of all elements.
  /// For complex arrays, computes mean of |z - mean|^2 (returns real).
  ///
  /// Uses a two-pass algorithm for numerical stability: the mean is computed
  /// first (via vDSP where applicable), then squared deviations are accumulated.
  ///
  /// - Parameter ddof: Delta degrees of freedom (default 0 for population variance).
  ///   Must be non-negative and strictly less than the number of elements.
  public func variance(ddof: Int = 0) -> Double {
    precondition(ddof >= 0, "ddof must be non-negative")
    precondition(ddof < size, "ddof (\(ddof)) must be less than array size (\(size))")

    if isComplex, let imagPart = imag {
      // Pass 1: compute complex mean via vDSP
      let meanVal = complexMean()
      // Pass 2: accumulate squared deviations from mean
      var sumSq: Double = 0
      for i in 0..<size {
        let diffReal = real[i] - meanVal.real
        let diffImag = imagPart[i] - meanVal.imag
        sumSq += diffReal * diffReal + diffImag * diffImag
      }
      return sumSq / Double(size - ddof)
    }

    // Pass 1: compute mean via vDSP
    let m = mean()
    // Pass 2: accumulate squared deviations from mean
    var sumSq: Double = 0
    for i in 0..<size {
      let diff = real[i] - m
      sumSq += diff * diff
    }
    return sumSq / Double(size - ddof)
  }

  /// Variance along an axis.
  /// For complex arrays, computes mean of |z - mean|^2 (returns real).
  ///
  /// Uses a two-pass algorithm for numerical stability: the slice mean is computed
  /// first, then squared deviations are accumulated.
  ///
  /// - Parameters:
  ///   - axis: Axis along which to compute variance.
  ///   - ddof: Delta degrees of freedom (default 0 for population variance).
  ///     Must be non-negative and strictly less than the slice length along `axis`.
  public func variance(axis: Int, ddof: Int = 0) -> NDArray {
    precondition(ddof >= 0, "ddof must be non-negative")
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      preconditionFailure("Axis \(axis) out of bounds for array with \(ndim) dimensions")
    }
    let axisLen = shape[normalizedAxis]
    precondition(
      ddof < axisLen,
      "ddof (\(ddof)) must be less than axis length (\(axisLen))")

    if isComplex {
      return reduceAlongAxisComplexToReal(axis: axis) { base, axisStride, count in
        let imagPart = imag!
        // Pass 1: compute complex mean
        var realSum = 0.0
        var imagSum = 0.0
        for i in 0..<count {
          realSum += real[base + i * axisStride]
          imagSum += imagPart[base + i * axisStride]
        }
        let n = Double(count)
        let meanReal = realSum / n
        let meanImag = imagSum / n
        // Pass 2: accumulate squared deviations from mean
        var sumSq = 0.0
        for i in 0..<count {
          let idx = base + i * axisStride
          let diffReal = real[idx] - meanReal
          let diffImag = imagPart[idx] - meanImag
          sumSq += diffReal * diffReal + diffImag * diffImag
        }
        return sumSq / Double(count - ddof)
      }
    }
    return reduceAlongAxis(axis: axis) { base, axisStride, count in
      // Pass 1: compute mean
      var sum = 0.0
      for i in 0..<count { sum += real[base + i * axisStride] }
      let m = sum / Double(count)
      // Pass 2: accumulate squared deviations from mean
      var sumSq = 0.0
      for i in 0..<count {
        let diff = real[base + i * axisStride] - m
        sumSq += diff * diff
      }
      return sumSq / Double(count - ddof)
    }
  }

  /// Standard deviation of all elements.
  /// For complex arrays, returns sqrt of variance (real result).
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
    reduceAlongAxis(axis: axis) { base, axisStride, count in
      var result = real[base]
      for i in 1..<count {
        let v = real[base + i * axisStride]
        if v < result { result = v }
      }
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
    reduceAlongAxis(axis: axis) { base, axisStride, count in
      var result = real[base]
      for i in 1..<count {
        let v = real[base + i * axisStride]
        if v > result { result = v }
      }
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
    if isComplex, let imagPart = imag {
      var resultReal = [Double](repeating: 0, count: size)
      var resultImag = [Double](repeating: 0, count: size)
      var sumReal: Double = 0
      var sumImag: Double = 0
      for i in 0..<size {
        sumReal += real[i]
        sumImag += imagPart[i]
        resultReal[i] = sumReal
        resultImag[i] = sumImag
      }
      return NDArray.complexArray(shape: [size], real: resultReal, imag: resultImag)
    }

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
    if isComplex, let imagPart = imag {
      var resultReal = [Double](repeating: 0, count: size)
      var resultImag = [Double](repeating: 0, count: size)
      var prodReal: Double = 1
      var prodImag: Double = 0
      for i in 0..<size {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let a = prodReal
        let b = prodImag
        let c = real[i]
        let d = imagPart[i]
        prodReal = a * c - b * d
        prodImag = a * d + b * c
        resultReal[i] = prodReal
        resultImag[i] = prodImag
      }
      return NDArray.complexArray(shape: [size], real: resultReal, imag: resultImag)
    }

    var result = [Double](repeating: 0, count: size)
    var prod: Double = 1
    for i in 0..<size {
      prod *= real[i]
      result[i] = prod
    }
    return NDArray(shape: [size], data: result)
  }


  /// Cumulative sum along an axis, preserving the input shape.
  ///
  /// Each element along `axis` is replaced by the running sum up to that position.
  /// For complex arrays both real and imaginary parts are accumulated independently.
  ///
  /// - Parameter axis: Axis along which to accumulate (supports negative indexing).
  /// - Returns: Array with the same shape as `self`.
  public func cumsum(axis: Int) -> NDArray {
    let ax = normalizeAxis(axis)
    let axisStride = strides[ax]
    let axisLen = shape[ax]

    var resultReal = real
    var resultImag = imag

    // Prefix-scan: for step k, add element at (k-1) into element at k.
    for k in 1..<axisLen {
      for i in 0..<size {
        guard (i / axisStride) % axisLen == k else { continue }
        let prev = i - axisStride
        resultReal[i] += resultReal[prev]
        if resultImag != nil { resultImag![i] += resultImag![prev] }
      }
    }

    if isComplex, let ri = resultImag {
      return NDArray(shape: shape, dtype: .complex128, real: resultReal, imag: ri)
    }
    return NDArray(shape: shape, data: resultReal)
  }

  /// Cumulative product along an axis, preserving the input shape.
  ///
  /// Each element along `axis` is replaced by the running product up to that position.
  /// For complex arrays the full complex multiplication rule is applied at each step.
  ///
  /// - Parameter axis: Axis along which to accumulate (supports negative indexing).
  /// - Returns: Array with the same shape as `self`.
  public func cumprod(axis: Int) -> NDArray {
    let ax = normalizeAxis(axis)
    let axisStride = strides[ax]
    let axisLen = shape[ax]

    var resultReal = real
    var resultImag = imag

    for k in 1..<axisLen {
      for i in 0..<size {
        guard (i / axisStride) % axisLen == k else { continue }
        let prev = i - axisStride
        if isComplex, resultImag != nil {
          // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
          let a = resultReal[prev], b = resultImag![prev]
          let c = resultReal[i], d = resultImag![i]
          resultReal[i] = a * c - b * d
          resultImag![i] = a * d + b * c
        } else {
          resultReal[i] *= resultReal[prev]
        }
      }
    }

    if isComplex, let ri = resultImag {
      return NDArray(shape: shape, dtype: .complex128, real: resultReal, imag: ri)
    }
    return NDArray(shape: shape, data: resultReal)
  }

  /// Differences along an axis.
  ///
  /// Computes `n`-th order differences along `axis`. The output shape is the same as
  /// the input except the `axis` dimension is reduced by `n`.
  ///
  /// - Parameters:
  ///   - n: Order of differences (default 1).
  ///   - axis: Axis along which to compute differences (supports negative indexing).
  /// - Returns: Array whose `axis` dimension is `shape[axis] - n`.
  public func diff(n: Int = 1, axis: Int) -> NDArray {
    let ax = normalizeAxis(axis)
    guard shape[ax] > n else {
      var zeroShape = shape
      zeroShape[ax] = 0
      if isComplex {
        return NDArray(shape: zeroShape, dtype: .complex128, real: [], imag: [])
      }
      return NDArray(shape: zeroShape, data: [])
    }

    var currentShape = shape
    var currentReal = real
    var currentImag = imag

    for _ in 0..<n {
      let curAxisLen = currentShape[ax]
      var newShape = currentShape
      newShape[ax] = curAxisLen - 1
      let newSize = newShape.reduce(1, *)

      var curStrides = [Int](repeating: 1, count: currentShape.count)
      for d in stride(from: currentShape.count - 2, through: 0, by: -1) {
        curStrides[d] = curStrides[d + 1] * currentShape[d + 1]
      }
      var newStrides = [Int](repeating: 1, count: newShape.count)
      for d in stride(from: newShape.count - 2, through: 0, by: -1) {
        newStrides[d] = newStrides[d + 1] * newShape[d + 1]
      }

      var newReal = [Double](repeating: 0, count: newSize)
      var newImag: [Double]? = currentImag != nil ? [Double](repeating: 0, count: newSize) : nil

      for outIdx in 0..<newSize {
        // Decode multi-dim coordinates of the output index.
        var coords = [Int](repeating: 0, count: newShape.count)
        var rem = outIdx
        for d in 0..<newShape.count {
          coords[d] = rem / newStrides[d]
          rem %= newStrides[d]
        }
        // Map to two consecutive source flat indices along ax.
        var srcIdx0 = 0, srcIdx1 = 0
        for d in 0..<currentShape.count {
          srcIdx0 += coords[d] * curStrides[d]
          srcIdx1 += (d == ax ? coords[d] + 1 : coords[d]) * curStrides[d]
        }
        newReal[outIdx] = currentReal[srcIdx1] - currentReal[srcIdx0]
        if newImag != nil, let ci = currentImag {
          newImag![outIdx] = ci[srcIdx1] - ci[srcIdx0]
        }
      }

      currentShape = newShape
      currentReal = newReal
      currentImag = newImag
    }

    if isComplex, let ri = currentImag {
      return NDArray(shape: currentShape, dtype: .complex128, real: currentReal, imag: ri)
    }
    return NDArray(shape: currentShape, data: currentReal)
  }

  /// Differences between consecutive elements.
  public func diff(n: Int = 1) -> NDArray {
    guard size > n else {
      if isComplex {
        return NDArray(shape: [0], dtype: .complex128, real: [], imag: [])
      }
      return NDArray(shape: [0], data: [])
    }

    if isComplex, let imagPart = imag {
      var currentReal = real
      var currentImag = imagPart
      for _ in 0..<n {
        var newReal = [Double](repeating: 0, count: currentReal.count - 1)
        var newImag = [Double](repeating: 0, count: currentImag.count - 1)
        for i in 0..<newReal.count {
          newReal[i] = currentReal[i + 1] - currentReal[i]
          newImag[i] = currentImag[i + 1] - currentImag[i]
        }
        currentReal = newReal
        currentImag = newImag
      }
      return NDArray.complexArray(shape: [currentReal.count], real: currentReal, imag: currentImag)
    }

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
    reduceAlongAxis(axis: axis) { base, axisStride, count in
      for i in 0..<count {
        if real[base + i * axisStride] == 0 { return 0 }
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
    reduceAlongAxis(axis: axis) { base, axisStride, count in
      for i in 0..<count {
        if real[base + i * axisStride] != 0 { return 1 }
      }
      return 0
    }
  }

  // MARK: - Tolerance Comparison

  /// Returns true if two arrays are element-wise equal within given tolerances.
  ///
  /// Follows NumPy's convention: `|a - b| <= atol + rtol * |b|` for every element.
  /// For complex arrays both the real and imaginary parts must satisfy the condition.
  ///
  /// - Parameters:
  ///   - a: First array.
  ///   - b: Second array.
  ///   - rtol: Relative tolerance (default `1e-5`).
  ///   - atol: Absolute tolerance (default `1e-8`).
  /// - Returns: `true` when all element-wise differences are within tolerance.
  public static func allclose(
    _ a: NDArray,
    _ b: NDArray,
    rtol: Double = 1e-5,
    atol: Double = 1e-8
  ) -> Bool {
    guard a.shape == b.shape else { return false }
    let n = a.size

    for i in 0..<n {
      let diff = Swift.abs(a.real[i] - b.real[i])
      let tol = atol + rtol * Swift.abs(b.real[i])
      if diff > tol { return false }
    }

    if a.isComplex || b.isComplex {
      let aImag = a.imag ?? [Double](repeating: 0, count: n)
      let bImag = b.imag ?? [Double](repeating: 0, count: n)
      for i in 0..<n {
        let diff = Swift.abs(aImag[i] - bImag[i])
        let tol = atol + rtol * Swift.abs(bImag[i])
        if diff > tol { return false }
      }
    }

    return true
  }

  // MARK: - Statistics

  /// Median value.
  public func median() -> Double {
    let sorted = real.sorted()
    let n = sorted.count
    if n == 0 { return .nan }
    if n % 2 == 0 {
      return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    } else {
      return sorted[n / 2]
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

  // MARK: - Axis Reduction Helpers

  /// Compute the flat base index into the source array for a given linear result index.
  ///
  /// Translates a linear index into the result array back to a flat index in the
  /// source array pointing to the first element of the slice along the reduced axis.
  ///
  /// - Parameters:
  ///   - resultIdx: Linear index into the result array.
  ///   - normalizedAxis: The axis being reduced (already validated and non-negative).
  ///   - resultShape: Shape of the result array (reduced axis removed).
  ///   - srcStrides: Row-major strides of the source array.
  /// - Returns: Flat index into the source array at the start of the axis slice.
  private func baseIndex(
    resultIdx: Int,
    normalizedAxis: Int,
    resultShape: [Int],
    srcStrides: [Int]
  ) -> Int {
    var baseIdx = 0
    var remaining = resultIdx
    var resultDim = 0
    for d in 0..<ndim {
      guard d != normalizedAxis else { continue }
      var resultStride = 1
      for rd in (resultDim + 1)..<resultShape.count {
        resultStride *= resultShape[rd]
      }
      let coord = remaining / resultStride
      remaining = remaining % resultStride
      baseIdx += coord * srcStrides[d]
      resultDim += 1
    }
    return baseIdx
  }

  /// Apply a reduction function along an axis for complex arrays.
  ///
  /// The closure receives `(base, axisStride, count)` and must read directly from
  /// `self.real` and `self.imag` — no intermediate slice is allocated.
  private func reduceAlongAxisComplex(
    axis: Int,
    reducer: (_ base: Int, _ axisStride: Int, _ count: Int) -> (Double, Double)
  ) -> NDArray {
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for array with \(ndim) dimensions")
    }

    var resultShape = shape
    resultShape.remove(at: normalizedAxis)
    if resultShape.isEmpty { resultShape = [1] }

    let resultSize = resultShape.reduce(1, *)
    var resultReal = [Double](repeating: 0, count: resultSize)
    var resultImag = [Double](repeating: 0, count: resultSize)

    var srcStrides = [Int](repeating: 1, count: ndim)
    for i in stride(from: ndim - 2, through: 0, by: -1) {
      srcStrides[i] = srcStrides[i + 1] * shape[i + 1]
    }

    let axisSize = shape[normalizedAxis]
    let axisStride = srcStrides[normalizedAxis]

    for resultIdx in 0..<resultSize {
      let base = baseIndex(
        resultIdx: resultIdx,
        normalizedAxis: normalizedAxis,
        resultShape: resultShape,
        srcStrides: srcStrides)
      let (r, im) = reducer(base, axisStride, axisSize)
      resultReal[resultIdx] = r
      resultImag[resultIdx] = im
    }

    return NDArray(shape: resultShape, dtype: .complex128, real: resultReal, imag: resultImag)
  }

  /// Apply a reduction function along an axis for complex arrays, returning a real result.
  ///
  /// The closure receives `(base, axisStride, count)` and must read directly from
  /// `self.real` and `self.imag` — no intermediate slice is allocated.
  private func reduceAlongAxisComplexToReal(
    axis: Int,
    reducer: (_ base: Int, _ axisStride: Int, _ count: Int) -> Double
  ) -> NDArray {
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for array with \(ndim) dimensions")
    }

    var resultShape = shape
    resultShape.remove(at: normalizedAxis)
    if resultShape.isEmpty { resultShape = [1] }

    let resultSize = resultShape.reduce(1, *)
    var resultData = [Double](repeating: 0, count: resultSize)

    var srcStrides = [Int](repeating: 1, count: ndim)
    for i in stride(from: ndim - 2, through: 0, by: -1) {
      srcStrides[i] = srcStrides[i + 1] * shape[i + 1]
    }

    let axisSize = shape[normalizedAxis]
    let axisStride = srcStrides[normalizedAxis]

    for resultIdx in 0..<resultSize {
      let base = baseIndex(
        resultIdx: resultIdx,
        normalizedAxis: normalizedAxis,
        resultShape: resultShape,
        srcStrides: srcStrides)
      resultData[resultIdx] = reducer(base, axisStride, axisSize)
    }

    return NDArray(shape: resultShape, data: resultData)
  }

  /// Apply a reduction function along an axis.
  ///
  /// The closure receives `(base, axisStride, count)` and must read directly from
  /// `self.real` — no intermediate slice is allocated.
  private func reduceAlongAxis(
    axis: Int,
    reducer: (_ base: Int, _ axisStride: Int, _ count: Int) -> Double
  ) -> NDArray {
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for array with \(ndim) dimensions")
    }

    var resultShape = shape
    resultShape.remove(at: normalizedAxis)
    if resultShape.isEmpty { resultShape = [1] }

    let resultSize = resultShape.reduce(1, *)
    var resultData = [Double](repeating: 0, count: resultSize)

    var srcStrides = [Int](repeating: 1, count: ndim)
    for i in stride(from: ndim - 2, through: 0, by: -1) {
      srcStrides[i] = srcStrides[i + 1] * shape[i + 1]
    }

    let axisSize = shape[normalizedAxis]
    let axisStride = srcStrides[normalizedAxis]

    for resultIdx in 0..<resultSize {
      let base = baseIndex(
        resultIdx: resultIdx,
        normalizedAxis: normalizedAxis,
        resultShape: resultShape,
        srcStrides: srcStrides)
      resultData[resultIdx] = reducer(base, axisStride, axisSize)
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

/// Cumulative sum along an axis.
public func cumsum(_ array: NDArray, axis: Int) -> NDArray { array.cumsum(axis: axis) }

/// Cumulative product.
public func cumprod(_ array: NDArray) -> NDArray { array.cumprod() }

/// Cumulative product along an axis.
public func cumprod(_ array: NDArray, axis: Int) -> NDArray { array.cumprod(axis: axis) }

/// Differences between consecutive elements.
public func diff(_ array: NDArray, n: Int = 1) -> NDArray { array.diff(n: n) }

/// Differences along an axis.
public func diff(_ array: NDArray, n: Int = 1, axis: Int) -> NDArray { array.diff(n: n, axis: axis) }

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

/// Returns true if two arrays are element-wise equal within given tolerances.
///
/// Follows NumPy's convention: `|a - b| <= atol + rtol * |b|` for every element.
/// For complex arrays both the real and imaginary parts must satisfy the condition.
///
/// - Parameters:
///   - a: First array.
///   - b: Second array.
///   - rtol: Relative tolerance (default `1e-5`).
///   - atol: Absolute tolerance (default `1e-8`).
/// - Returns: `true` when all element-wise differences are within tolerance.
public func allclose(
  _ a: NDArray,
  _ b: NDArray,
  rtol: Double = 1e-5,
  atol: Double = 1e-8
) -> Bool {
  NDArray.allclose(a, b, rtol: rtol, atol: atol)
}
