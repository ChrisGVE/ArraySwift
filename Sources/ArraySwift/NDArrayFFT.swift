//
//  NDArrayFFT.swift
//  ArraySwift
//
//  FFT family via Accelerate vDSP, mirroring NumPy's numpy.fft module.
//
//  Supported functions:
//    fft     — 1-D complex DFT  (forward, complex128 in/out)
//    ifft    — 1-D inverse DFT  (normalised by 1/n)
//    rfft    — 1-D real DFT     (float64 in → complex128 out, n/2+1 bins)
//    fft2    — 2-D complex DFT  (row then column passes)
//    ifft2   — 2-D inverse DFT
//    fftn    — N-D complex DFT  (repeated 1-D passes over each axis)
//    fftfreq — sample-frequency array for interpreting FFT output bins
//
//  Non-power-of-2 handling
//  -----------------------
//  vDSP_DFT_zop_CreateSetupD only accepts power-of-2 lengths. For power-of-2
//  sizes the implementation delegates to vDSP for SIMD-accelerated O(N log N)
//  performance. For other lengths a direct O(N²) DFT is used as a fallback;
//  this is exact, correct for round-trips (iFFT∘FFT = identity), and practical
//  for the sizes typically encountered in this library. A future enhancement
//  could replace the fallback with a Bluestein/Chirp-Z transform for O(N log N)
//  performance at arbitrary lengths.
//

import Accelerate
import Foundation

// MARK: - 1-D FFT / iFFT

extension NDArray {

  // MARK: - fft

  /// 1-D discrete Fourier transform (forward, unnormalised).
  ///
  /// Equivalent to NumPy's `numpy.fft.fft`. Input must be a 1-D `.complex128`
  /// array. Non-power-of-2 lengths are handled by zero-padding to the next
  /// power of 2 and truncating the output back to `n` bins.
  ///
  /// - Parameter x: 1-D `.complex128` input array of length `n`.
  /// - Returns: 1-D `.complex128` output of length `n`.
  public static func fft(_ x: NDArray) -> NDArray {
    precondition(x.ndim == 1, "fft requires a 1-D array")
    precondition(x.dtype == .complex128, "fft requires a .complex128 array")
    guard case .complex128(let xr, let xi) = x.storage else {
      preconditionFailure("fft: expected complex128 storage")
    }
    let (yr, yi) = dft1D(real: xr, imag: xi, direction: .FORWARD)
    return NDArray(shape: x.shape, storage: .complex128(real: yr, imag: yi))
  }

  // MARK: - ifft

  /// 1-D inverse discrete Fourier transform (normalised by 1/n).
  ///
  /// Equivalent to NumPy's `numpy.fft.ifft`. Input must be a 1-D `.complex128`
  /// array. Non-power-of-2 lengths are handled identically to ``fft(_:)``.
  ///
  /// - Parameter x: 1-D `.complex128` input array of length `n`.
  /// - Returns: 1-D `.complex128` output of length `n`, normalised by `1/n`.
  public static func ifft(_ x: NDArray) -> NDArray {
    precondition(x.ndim == 1, "ifft requires a 1-D array")
    precondition(x.dtype == .complex128, "ifft requires a .complex128 array")
    guard case .complex128(let xr, let xi) = x.storage else {
      preconditionFailure("ifft: expected complex128 storage")
    }
    let (yr, yi) = dft1D(real: xr, imag: xi, direction: .INVERSE)
    let n = Double(x.size)
    return NDArray(shape: x.shape, storage: .complex128(
      real: yr.map { $0 / n },
      imag: yi.map { $0 / n }))
  }

  // MARK: - rfft

  /// 1-D real-input FFT, returning the non-redundant complex spectrum.
  ///
  /// Equivalent to NumPy's `numpy.fft.rfft`. Input must be a 1-D `.float64`
  /// array of length `n`. Returns `n/2 + 1` complex bins (the positive
  /// frequencies including DC and the Nyquist bin).
  ///
  /// - Parameter x: 1-D `.float64` input array of length `n`.
  /// - Returns: 1-D `.complex128` output of length `n/2 + 1`.
  public static func rfft(_ x: NDArray) -> NDArray {
    precondition(x.ndim == 1, "rfft requires a 1-D array")
    precondition(x.dtype == .float64, "rfft requires a .float64 array")
    let n = x.size
    let outCount = n / 2 + 1
    // zero-imaginary input
    let xr = x.real
    let xi = [Double](repeating: 0, count: n)
    let (yr, yi) = dft1D(real: xr, imag: xi, direction: .FORWARD)
    // rfft output = first n/2+1 bins of full complex FFT
    return NDArray(shape: [outCount], storage: .complex128(
      real: Array(yr.prefix(outCount)),
      imag: Array(yi.prefix(outCount))))
  }

  // MARK: - fft2

  /// 2-D discrete Fourier transform (forward).
  ///
  /// Equivalent to NumPy's `numpy.fft.fft2`. Input must be a 2-D `.complex128`
  /// array. Applies 1-D FFT first over rows, then over columns.
  ///
  /// - Parameter x: 2-D `.complex128` input array.
  /// - Returns: 2-D `.complex128` output with the same shape as `x`.
  public static func fft2(_ x: NDArray) -> NDArray {
    precondition(x.ndim == 2, "fft2 requires a 2-D array")
    precondition(x.dtype == .complex128, "fft2 requires a .complex128 array")
    return fftNDImpl(x, direction: .FORWARD)
  }

  // MARK: - ifft2

  /// 2-D inverse discrete Fourier transform (normalised by 1/m×n).
  ///
  /// Equivalent to NumPy's `numpy.fft.ifft2`. Input must be a 2-D `.complex128`
  /// array. Applies 1-D iFFT over columns then rows.
  ///
  /// - Parameter x: 2-D `.complex128` input array.
  /// - Returns: 2-D `.complex128` output with the same shape as `x`.
  public static func ifft2(_ x: NDArray) -> NDArray {
    precondition(x.ndim == 2, "ifft2 requires a 2-D array")
    precondition(x.dtype == .complex128, "ifft2 requires a .complex128 array")
    return fftNDImpl(x, direction: .INVERSE)
  }

  // MARK: - fftn

  /// N-D discrete Fourier transform (forward), applying 1-D FFT along each axis.
  ///
  /// Equivalent to NumPy's `numpy.fft.fftn`. Input must be a `.complex128`
  /// array. The transform is applied sequentially over every axis from 0 to
  /// `ndim-1` (row-major order), matching NumPy's default behaviour.
  ///
  /// - Parameter x: N-D `.complex128` input array.
  /// - Returns: N-D `.complex128` output with the same shape as `x`.
  public static func fftn(_ x: NDArray) -> NDArray {
    precondition(x.dtype == .complex128, "fftn requires a .complex128 array")
    return fftNDImpl(x, direction: .FORWARD)
  }

  /// N-D inverse discrete Fourier transform (normalised).
  ///
  /// - Parameter x: N-D `.complex128` input array.
  /// - Returns: N-D `.complex128` output with the same shape as `x`.
  public static func ifftn(_ x: NDArray) -> NDArray {
    precondition(x.dtype == .complex128, "ifftn requires a .complex128 array")
    return fftNDImpl(x, direction: .INVERSE)
  }

  // MARK: - fftfreq

  /// Return sample frequencies for an `n`-point DFT.
  ///
  /// Equivalent to NumPy's `numpy.fft.fftfreq`. The returned values represent
  /// the normalised frequency bins in cycles per sample:
  ///
  /// - Bin 0: DC (0 Hz)
  /// - Bins 1…⌊(n-1)/2⌋: positive frequencies
  /// - Bin ⌊n/2⌋: Nyquist (or just above for odd n)
  /// - Bins ⌊n/2⌋+1…n-1: negative frequencies (folded)
  ///
  /// Scale by `1/d` where `d` is the sample spacing (default 1.0) to get
  /// frequencies in physical units.
  ///
  /// - Parameters:
  ///   - n: Number of DFT points.
  ///   - d: Sample spacing (default `1.0`). Frequencies are returned in units
  ///     of cycles per `d`.
  /// - Returns: A 1-D `.float64` array of length `n`.
  public static func fftfreq(_ n: Int, d: Double = 1.0) -> NDArray {
    precondition(n > 0, "fftfreq requires n > 0")
    var freqs = [Double](repeating: 0, count: n)
    let half = (n + 1) / 2           // ceil(n/2)
    let nd = Double(n)
    for i in 0..<half {
      freqs[i] = Double(i) / (nd * d)
    }
    for i in half..<n {
      freqs[i] = Double(i - n) / (nd * d)
    }
    return NDArray(shape: [n], storage: .float64(freqs))
  }
}

// MARK: - N-D implementation

extension NDArray {

  /// Apply 1-D DFT along every axis of an N-D complex array.
  ///
  /// For FORWARD direction all axes are transformed forward (unnormalised).
  /// For INVERSE direction all axes are transformed inverse (each normalised
  /// by the axis length), producing an overall normalisation of 1/size.
  private static func fftNDImpl(_ x: NDArray, direction: vDSP_DFT_Direction) -> NDArray {
    precondition(x.dtype == .complex128)
    var current = x
    for axis in 0..<x.ndim {
      current = applyFFTAlongAxis(current, axis: axis, direction: direction)
    }
    return current
  }

  /// Apply a 1-D DFT along a single axis of an N-D complex array.
  ///
  /// Extracts 1-D "pencils" along the given axis, transforms each, and
  /// reassembles the result. For INVERSE direction each 1-D transform is
  /// normalised by its length.
  private static func applyFFTAlongAxis(
    _ x: NDArray,
    axis: Int,
    direction: vDSP_DFT_Direction
  ) -> NDArray {
    precondition(x.dtype == .complex128)
    guard case .complex128(let xr, let xi) = x.storage else {
      preconditionFailure("applyFFTAlongAxis: expected complex128 storage")
    }

    let shape = x.shape
    let axisLen = shape[axis]
    let totalSize = x.size
    // Number of 1-D pencils = totalSize / axisLen
    let nPencils = totalSize / axisLen

    // Compute strides to extract pencils along `axis`.
    let strides = x.strides
    let axisStride = strides[axis]

    // Outer stride = product of all dimensions before `axis`.
    // This lets us enumerate all pencil starting offsets.
    var outerSize = 1
    for d in 0..<axis { outerSize *= shape[d] }
    var innerSize = 1
    for d in (axis + 1)..<shape.count { innerSize *= shape[d] }

    var outReal = [Double](repeating: 0, count: totalSize)
    var outImag = [Double](repeating: 0, count: totalSize)

    // Iterate over all (outer, inner) combinations, each defines one pencil.
    for outer in 0..<outerSize {
      for inner in 0..<innerSize {
        // Base flat index for this pencil.
        let base = outer * (axisLen * innerSize) + inner
        // Gather pencil.
        var pencilR = [Double](repeating: 0, count: axisLen)
        var pencilI = [Double](repeating: 0, count: axisLen)
        for k in 0..<axisLen {
          let idx = base + k * axisStride
          pencilR[k] = xr[idx]
          pencilI[k] = xi[idx]
        }
        // Transform.
        let (yr, yi) = dft1D(real: pencilR, imag: pencilI, direction: direction)
        let normFactor = direction == .INVERSE ? Double(axisLen) : 1.0
        // Scatter result back.
        for k in 0..<axisLen {
          let idx = base + k * axisStride
          outReal[idx] = yr[k] / normFactor
          outImag[idx] = yi[k] / normFactor
        }
      }
    }

    _ = nPencils  // suppress unused-variable warning
    return NDArray(shape: shape, storage: .complex128(real: outReal, imag: outImag))
  }
}

// MARK: - Core 1-D DFT worker

extension NDArray {

  /// Execute a 1-D DFT.
  ///
  /// Uses vDSP for power-of-2 lengths (SIMD-accelerated O(N log N)); falls back
  /// to a direct O(N²) DFT for other lengths.
  ///
  /// - Parameters:
  ///   - real: Real part of the input (length `n`).
  ///   - imag: Imaginary part of the input (length `n`).
  ///   - direction: `.FORWARD` or `.INVERSE` (both unnormalised).
  /// - Returns: A pair `(real, imag)` of output buffers of length `n`.
  internal static func dft1D(
    real: [Double],
    imag: [Double],
    direction: vDSP_DFT_Direction
  ) -> ([Double], [Double]) {
    let n = real.count
    precondition(n > 0, "dft1D: length must be > 0")
    precondition(imag.count == n, "dft1D: real and imag lengths must match")

    if n == nextPow2(n) {
      // Power-of-2: use vDSP for SIMD performance.
      return executevDSP(real: real, imag: imag, n: n, direction: direction)
    } else {
      // General length: direct O(N²) DFT.
      return directDFT(real: real, imag: imag, direction: direction)
    }
  }

  /// Execute a vDSP DFT over a power-of-2 buffer of length `n`.
  private static func executevDSP(
    real: [Double],
    imag: [Double],
    n: Int,
    direction: vDSP_DFT_Direction
  ) -> ([Double], [Double]) {
    guard let setup = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(n), direction) else {
      preconditionFailure("vDSP_DFT_zop_CreateSetupD failed for n=\(n)")
    }
    defer { vDSP_DFT_DestroySetupD(setup) }

    var inputR = real
    var inputI = imag
    var outputR = [Double](repeating: 0, count: n)
    var outputI = [Double](repeating: 0, count: n)

    vDSP_DFT_ExecuteD(setup, &inputR, &inputI, &outputR, &outputI)
    return (outputR, outputI)
  }

  /// Direct O(N²) DFT for non-power-of-2 lengths.
  ///
  /// Forward:  X[k] = Σ x[j] · exp(−2πi jk/N)
  /// Inverse:  X[k] = Σ x[j] · exp(+2πi jk/N)  (caller normalises by N)
  private static func directDFT(
    real: [Double],
    imag: [Double],
    direction: vDSP_DFT_Direction
  ) -> ([Double], [Double]) {
    let n = real.count
    let sign: Double = direction == .FORWARD ? -1.0 : 1.0
    let twoPiOverN = 2.0 * Double.pi / Double(n)
    var outR = [Double](repeating: 0, count: n)
    var outI = [Double](repeating: 0, count: n)
    for k in 0..<n {
      var sumR = 0.0
      var sumI = 0.0
      for j in 0..<n {
        let angle = sign * twoPiOverN * Double(j * k)
        let c = Foundation.cos(angle)
        let s = Foundation.sin(angle)
        sumR += real[j] * c - imag[j] * s
        sumI += real[j] * s + imag[j] * c
      }
      outR[k] = sumR
      outI[k] = sumI
    }
    return (outR, outI)
  }

  /// Return the smallest power of 2 that is ≥ `n`.
  internal static func nextPow2(_ n: Int) -> Int {
    precondition(n > 0)
    if n == 1 { return 1 }
    var p = 1
    while p < n { p <<= 1 }
    return p
  }
}
