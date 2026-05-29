//
//  NDArrayFFTTests.swift
//  ArraySwiftTests
//
//  Tests for FFT family: fft, ifft, rfft, fft2, fftn, fftfreq.
//  Correctness validated against known transforms (NumPy reference values).
//

import XCTest
@testable import ArraySwift

final class NDArrayFFTTests: XCTestCase {

    let tol = 1e-9

    // MARK: - Helpers

    /// Assert complex NDArrays are close within tolerance.
    private func assertComplexClose(
        _ result: NDArray,
        real expectedReal: [Double],
        imag expectedImag: [Double],
        accuracy: Double = 1e-9,
        file: StaticString = #file, line: UInt = #line
    ) {
        XCTAssertEqual(result.dtype, .complex128, "Expected complex128", file: file, line: line)
        XCTAssertEqual(result.size, expectedReal.count, "Size mismatch", file: file, line: line)
        guard let resultImag = result.imag else {
            XCTFail("Missing imag", file: file, line: line); return
        }
        for i in 0..<expectedReal.count {
            XCTAssertEqual(result.real[i], expectedReal[i], accuracy: accuracy,
                           "real[\(i)] mismatch", file: file, line: line)
            XCTAssertEqual(resultImag[i], expectedImag[i], accuracy: accuracy,
                           "imag[\(i)] mismatch", file: file, line: line)
        }
    }

    // MARK: - fft (complex input)

    /// FFT of a delta function equals a constant: FFT([1,0,0,0]) = [1,1,1,1].
    func testFFTDelta() {
        let x = NDArray.complexArray(shape: [4], real: [1, 0, 0, 0], imag: [0, 0, 0, 0])
        let y = NDArray.fft(x)
        assertComplexClose(y, real: [1, 1, 1, 1], imag: [0, 0, 0, 0])
    }

    /// FFT of DC signal (all ones) gives spike at bin 0.
    func testFFTDC() {
        // [1,1,1,1] → [4, 0, 0, 0]
        let x = NDArray.complexArray(shape: [4], real: [1, 1, 1, 1], imag: [0, 0, 0, 0])
        let y = NDArray.fft(x)
        assertComplexClose(y, real: [4, 0, 0, 0], imag: [0, 0, 0, 0])
    }

    /// FFT of single-tone sinusoid: cos(2π k₀/N * n) peaks at bins k₀ and N-k₀.
    func testFFTSingleToneRealPart() {
        let n = 8
        let k0 = 2
        var re = [Double](repeating: 0, count: n)
        for i in 0..<n {
            re[i] = cos(2 * .pi * Double(k0 * i) / Double(n))
        }
        let x = NDArray.complexArray(shape: [n], real: re, imag: [Double](repeating: 0, count: n))
        let y = NDArray.fft(x)
        // bin k0 and n-k0 should have magnitude 4 (=n/2), others ~0
        for i in 0..<n {
            let mag = sqrt(y.real[i] * y.real[i] + (y.imag![i]) * (y.imag![i]))
            if i == k0 || i == n - k0 {
                XCTAssertEqual(mag, Double(n) / 2, accuracy: 1e-9, "bin \(i) magnitude")
            } else {
                XCTAssertEqual(mag, 0.0, accuracy: 1e-9, "bin \(i) should be ~0")
            }
        }
    }

    // MARK: - ifft

    /// iFFT(FFT(x)) ≈ x round-trip.
    func testIFFTRoundTrip() {
        let re: [Double] = [3, 1, 4, 1, 5, 9, 2, 6]
        let im: [Double] = [0, 0, 0, 0, 0, 0, 0, 0]
        let x = NDArray.complexArray(shape: [8], real: re, imag: im)
        let y = NDArray.fft(x)
        let z = NDArray.ifft(y)
        for i in 0..<8 {
            XCTAssertEqual(z.real[i], re[i], accuracy: 1e-9, "real[\(i)]")
            XCTAssertEqual(z.imag![i], im[i], accuracy: 1e-9, "imag[\(i)]")
        }
    }

    /// iFFT of constant = delta.
    func testIFFTOfConstant() {
        let x = NDArray.complexArray(shape: [4], real: [1, 1, 1, 1], imag: [0, 0, 0, 0])
        let z = NDArray.ifft(x)
        assertComplexClose(z, real: [1, 0, 0, 0], imag: [0, 0, 0, 0])
    }

    // MARK: - rfft (real input → complex output)

    /// rfft of real delta produces constant.
    func testRFFTDelta() {
        let x = NDArray([1.0, 0.0, 0.0, 0.0])
        let y = NDArray.rfft(x)
        // rfft output has n/2+1 = 3 elements
        XCTAssertEqual(y.size, 3)
        assertComplexClose(y, real: [1, 1, 1], imag: [0, 0, 0])
    }

    /// rfft round-trip via irfft is not required (no irfft implemented),
    /// but rfft of DC signal = [n, 0, 0, …].
    func testRFFTDCSignal() {
        let n = 8
        let x = NDArray([Double](repeating: 1, count: n))
        let y = NDArray.rfft(x)
        XCTAssertEqual(y.size, n / 2 + 1)
        XCTAssertEqual(y.real[0], Double(n), accuracy: 1e-9)
        for i in 1..<y.size {
            XCTAssertEqual(y.real[i], 0.0, accuracy: 1e-9, "real[\(i)]")
            XCTAssertEqual(y.imag![i], 0.0, accuracy: 1e-9, "imag[\(i)]")
        }
    }

    /// rfft requires a real (.float64) input array.
    func testRFFTRequiresRealInput() {
        let cx = NDArray.complexArray(shape: [4], real: [1, 0, 0, 0], imag: [0, 0, 0, 0])
        // Should trap or produce valid result for real part – here we test that
        // passing a float64 array works (the guard is for dtype, not complex)
        let x = NDArray([1.0, 0.0, 0.0, 0.0])
        let y = NDArray.rfft(x)
        XCTAssertEqual(y.dtype, .complex128)
    }

    // MARK: - fft2 (2-D)

    /// fft2 of all-ones 2×2 = [[4,0],[0,0]].
    func testFFT2Ones2x2() {
        let x = NDArray.complexArray(shape: [2, 2],
                                     real: [1, 1, 1, 1],
                                     imag: [0, 0, 0, 0])
        let y = NDArray.fft2(x)
        XCTAssertEqual(y.shape, [2, 2])
        XCTAssertEqual(y.real[0], 4.0, accuracy: 1e-9)    // (0,0)
        XCTAssertEqual(y.real[1], 0.0, accuracy: 1e-9)    // (0,1)
        XCTAssertEqual(y.real[2], 0.0, accuracy: 1e-9)    // (1,0)
        XCTAssertEqual(y.real[3], 0.0, accuracy: 1e-9)    // (1,1)
    }

    /// fft2 round-trip via ifft2.
    func testFFT2RoundTrip() {
        let re: [Double] = [1, 2, 3, 4]
        let im: [Double] = [0, 0, 0, 0]
        let x = NDArray.complexArray(shape: [2, 2], real: re, imag: im)
        let Y = NDArray.fft2(x)
        let z = NDArray.ifft2(Y)
        for i in 0..<4 {
            XCTAssertEqual(z.real[i], re[i], accuracy: 1e-9)
            XCTAssertEqual(z.imag![i], im[i], accuracy: 1e-9)
        }
    }

    // MARK: - fftn (N-D)

    /// fftn on a 1-D array is identical to fft.
    func testFFTnEqualsFFT1D() {
        let re: [Double] = [1, 2, 3, 4]
        let x = NDArray.complexArray(shape: [4], real: re, imag: [0, 0, 0, 0])
        let y1 = NDArray.fft(x)
        let yn = NDArray.fftn(x)
        XCTAssertEqual(y1.shape, yn.shape)
        for i in 0..<4 {
            XCTAssertEqual(y1.real[i], yn.real[i], accuracy: 1e-9)
            XCTAssertEqual(y1.imag![i], yn.imag![i], accuracy: 1e-9)
        }
    }

    /// fftn on 2-D equals fft2.
    func testFFTnEqualsFFT2D() {
        let re: [Double] = [1, 2, 3, 4]
        let x = NDArray.complexArray(shape: [2, 2], real: re, imag: [0, 0, 0, 0])
        let y2 = NDArray.fft2(x)
        let yn = NDArray.fftn(x)
        for i in 0..<4 {
            XCTAssertEqual(y2.real[i], yn.real[i], accuracy: 1e-9)
            XCTAssertEqual(y2.imag![i], yn.imag![i], accuracy: 1e-9)
        }
    }

    // MARK: - fftfreq

    /// fftfreq(4) == [0.0, 0.25, -0.5, -0.25] (NumPy reference).
    func testFFTFreq4() {
        let f = NDArray.fftfreq(4)
        XCTAssertEqual(f.size, 4)
        XCTAssertEqual(f.real[0],  0.00, accuracy: 1e-12)
        XCTAssertEqual(f.real[1],  0.25, accuracy: 1e-12)
        XCTAssertEqual(f.real[2], -0.50, accuracy: 1e-12)
        XCTAssertEqual(f.real[3], -0.25, accuracy: 1e-12)
    }

    /// fftfreq(5) == [0.0, 0.2, 0.4, -0.4, -0.2] (odd length, NumPy reference).
    func testFFTFreq5() {
        let f = NDArray.fftfreq(5)
        XCTAssertEqual(f.size, 5)
        XCTAssertEqual(f.real[0],  0.0, accuracy: 1e-12)
        XCTAssertEqual(f.real[1],  0.2, accuracy: 1e-12)
        XCTAssertEqual(f.real[2],  0.4, accuracy: 1e-12)
        XCTAssertEqual(f.real[3], -0.4, accuracy: 1e-12)
        XCTAssertEqual(f.real[4], -0.2, accuracy: 1e-12)
    }

    /// fftfreq with sample spacing d=0.5 scales all frequencies by 1/d.
    func testFFTFreqWithSpacing() {
        let f = NDArray.fftfreq(4, d: 0.5)
        XCTAssertEqual(f.real[0],  0.0, accuracy: 1e-12)
        XCTAssertEqual(f.real[1],  0.5, accuracy: 1e-12)
        XCTAssertEqual(f.real[2], -1.0, accuracy: 1e-12)
        XCTAssertEqual(f.real[3], -0.5, accuracy: 1e-12)
    }

    // MARK: - Non-power-of-2 lengths (zero-padding)

    /// FFT of 5-element delta; result should have 5 elements with all magnitude 1.
    func testFFTNonPow2Length() {
        let re: [Double] = [1, 0, 0, 0, 0]
        let x = NDArray.complexArray(shape: [5], real: re, imag: [0, 0, 0, 0, 0])
        let y = NDArray.fft(x)
        XCTAssertEqual(y.size, 5, "Output must match input length")
        for i in 0..<5 {
            let mag = sqrt(y.real[i] * y.real[i] + y.imag![i] * y.imag![i])
            XCTAssertEqual(mag, 1.0, accuracy: 1e-9, "bin \(i) magnitude")
        }
    }

    /// iFFT(FFT(x)) round-trip for non-power-of-2 length.
    func testFFTRoundTripNonPow2() {
        let re: [Double] = [1, 2, 3, 4, 5]
        let x = NDArray.complexArray(shape: [5], real: re, imag: [0, 0, 0, 0, 0])
        let y = NDArray.fft(x)
        let z = NDArray.ifft(y)
        XCTAssertEqual(z.size, 5)
        for i in 0..<5 {
            XCTAssertEqual(z.real[i], re[i], accuracy: 1e-9, "real[\(i)]")
        }
    }
}
