//
//  NDArrayTests.swift
//  ArraySwiftTests
//
//  Tests for NDArray functionality
//

import XCTest
@testable import ArraySwift

final class NDArrayTests: XCTestCase {

    let epsilon = 1e-10

    // MARK: - Creation Tests

    func testZeros() {
        let arr = NDArray.zeros([2, 3])
        XCTAssertEqual(arr.shape, [2, 3])
        XCTAssertEqual(arr.size, 6)
        XCTAssertTrue(arr.real.allSatisfy { $0 == 0 })
    }

    func testOnes() {
        let arr = NDArray.ones([3, 2])
        XCTAssertEqual(arr.shape, [3, 2])
        XCTAssertTrue(arr.real.allSatisfy { $0 == 1 })
    }

    func testFull() {
        let arr = NDArray.full([2, 2], value: 5.0)
        XCTAssertEqual(arr.shape, [2, 2])
        XCTAssertTrue(arr.real.allSatisfy { $0 == 5 })
    }

    func testArange() {
        let arr = NDArray.arange(start: 0, stop: 5, step: 1)
        XCTAssertEqual(arr.shape, [5])
        XCTAssertEqual(arr.real, [0, 1, 2, 3, 4])
    }

    func testArangeWithStep() {
        let arr = NDArray.arange(start: 0, stop: 10, step: 2)
        XCTAssertEqual(arr.real, [0, 2, 4, 6, 8])
    }

    func testLinspace() {
        let arr = NDArray.linspace(start: 0, stop: 1, num: 5)
        XCTAssertEqual(arr.size, 5)
        XCTAssertEqual(arr.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(arr.real[4], 1, accuracy: epsilon)
        XCTAssertEqual(arr.real[2], 0.5, accuracy: epsilon)
    }

    func testEye() {
        let arr = NDArray.eye(3)
        XCTAssertEqual(arr.shape, [3, 3])
        XCTAssertEqual(arr.real[0], 1)  // [0,0]
        XCTAssertEqual(arr.real[4], 1)  // [1,1]
        XCTAssertEqual(arr.real[8], 1)  // [2,2]
        XCTAssertEqual(arr.real[1], 0)  // [0,1]
    }

    func testDiag() {
        let values = NDArray([1.0, 2.0, 3.0])
        let diag = NDArray.diag(values)
        XCTAssertEqual(diag.shape, [3, 3])
        XCTAssertEqual(diag.real[0], 1)  // [0,0]
        XCTAssertEqual(diag.real[4], 2)  // [1,1]
        XCTAssertEqual(diag.real[8], 3)  // [2,2]
    }

    func testRandom() {
        let arr = NDArray.random([100])
        XCTAssertEqual(arr.size, 100)
        XCTAssertTrue(arr.real.allSatisfy { $0 >= 0 && $0 < 1 })
    }

    func testRandn() {
        let arr = NDArray.randn([1000])
        XCTAssertEqual(arr.size, 1000)
        // Check mean is approximately 0
        let mean = arr.mean()
        XCTAssertEqual(mean, 0, accuracy: 0.2)
    }

    // MARK: - Arithmetic Tests

    func testAdd() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([4.0, 5.0, 6.0])
        let c = a.add(b)
        XCTAssertEqual(c.real, [5, 7, 9])
    }

    func testSubtract() {
        let a = NDArray([5.0, 6.0, 7.0])
        let b = NDArray([1.0, 2.0, 3.0])
        let c = a.subtract(b)
        XCTAssertEqual(c.real, [4, 4, 4])
    }

    func testMultiply() {
        let a = NDArray([2.0, 3.0, 4.0])
        let b = NDArray([3.0, 4.0, 5.0])
        let c = a.multiply(b)
        XCTAssertEqual(c.real, [6, 12, 20])
    }

    func testDivide() {
        let a = NDArray([10.0, 20.0, 30.0])
        let b = NDArray([2.0, 4.0, 5.0])
        let c = a.divide(b)
        XCTAssertEqual(c.real, [5, 5, 6])
    }

    func testOperators() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([4.0, 5.0, 6.0])

        XCTAssertEqual((a + b).real, [5, 7, 9])
        XCTAssertEqual((b - a).real, [3, 3, 3])
        XCTAssertEqual((a * b).real, [4, 10, 18])
    }

    func testScalarOperations() {
        let a = NDArray([1.0, 2.0, 3.0])

        XCTAssertEqual((a + 10).real, [11, 12, 13])
        XCTAssertEqual((a * 2).real, [2, 4, 6])
        XCTAssertEqual((10 + a).real, [11, 12, 13])
        XCTAssertEqual((2 * a).real, [2, 4, 6])
    }

    func testNegation() {
        let a = NDArray([1.0, -2.0, 3.0])
        let b = -a
        XCTAssertEqual(b.real, [-1, 2, -3])
    }

    func testPower() {
        let a = NDArray([2.0, 3.0, 4.0])
        let b = a.power(2)
        XCTAssertEqual(b.real, [4, 9, 16])
    }

    // MARK: - Math Function Tests

    func testAbs() {
        let a = NDArray([-1.0, 2.0, -3.0])
        let b = a.abs()
        XCTAssertEqual(b.real, [1, 2, 3])
    }

    func testSqrt() {
        let a = NDArray([1.0, 4.0, 9.0, 16.0])
        let b = a.sqrt()
        XCTAssertEqual(b.real[0], 1, accuracy: epsilon)
        XCTAssertEqual(b.real[1], 2, accuracy: epsilon)
        XCTAssertEqual(b.real[2], 3, accuracy: epsilon)
        XCTAssertEqual(b.real[3], 4, accuracy: epsilon)
    }

    func testExp() {
        let a = NDArray([0.0, 1.0])
        let b = a.exp()
        XCTAssertEqual(b.real[0], 1, accuracy: epsilon)
        XCTAssertEqual(b.real[1], Darwin.exp(1), accuracy: epsilon)
    }

    func testLog() {
        let a = NDArray([1.0, Darwin.exp(1)])
        let b = a.log()
        XCTAssertEqual(b.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(b.real[1], 1, accuracy: epsilon)
    }

    func testLog10() {
        let a = NDArray([1.0, 10.0, 100.0])
        let b = a.log10()
        XCTAssertEqual(b.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(b.real[1], 1, accuracy: epsilon)
        XCTAssertEqual(b.real[2], 2, accuracy: epsilon)
    }

    func testSinCos() {
        let a = NDArray([0.0, .pi / 2, .pi])
        let sinA = a.sin()
        let cosA = a.cos()

        XCTAssertEqual(sinA.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(sinA.real[1], 1, accuracy: epsilon)
        XCTAssertEqual(sinA.real[2], 0, accuracy: epsilon)

        XCTAssertEqual(cosA.real[0], 1, accuracy: epsilon)
        XCTAssertEqual(cosA.real[1], 0, accuracy: epsilon)
        XCTAssertEqual(cosA.real[2], -1, accuracy: epsilon)
    }

    func testTan() {
        let a = NDArray([0.0, .pi / 4])
        let b = a.tan()
        XCTAssertEqual(b.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(b.real[1], 1, accuracy: epsilon)
    }

    func testFloorCeil() {
        let a = NDArray([1.2, 2.8, -1.5])
        XCTAssertEqual(a.floor().real, [1, 2, -2])
        XCTAssertEqual(a.ceil().real, [2, 3, -1])
    }

    func testTrunc() {
        let a = NDArray([1.9, -1.9, 2.1])
        XCTAssertEqual(a.trunc().real, [1, -1, 2])
    }

    func testRound() {
        let a = NDArray([1.4, 1.5, 1.6, -1.5])
        let b = a.round()
        XCTAssertEqual(b.real[0], 1, accuracy: epsilon)
        XCTAssertEqual(b.real[1], 2, accuracy: epsilon)
        XCTAssertEqual(b.real[2], 2, accuracy: epsilon)
    }

    func testClip() {
        let a = NDArray([1.0, 5.0, 10.0, 15.0])
        let b = a.clip(min: 3, max: 12)
        XCTAssertEqual(b.real, [3, 5, 10, 12])
    }

    // MARK: - Reduction Tests

    func testSum() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        XCTAssertEqual(a.sum(), 10)
    }

    func testSumAxis() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let sumAxis0 = a.sum(axis: 0)
        let sumAxis1 = a.sum(axis: 1)

        XCTAssertEqual(sumAxis0.shape, [3])
        XCTAssertEqual(sumAxis0.real, [5, 7, 9])

        XCTAssertEqual(sumAxis1.shape, [2])
        XCTAssertEqual(sumAxis1.real, [6, 15])
    }

    func testProd() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        XCTAssertEqual(a.prod(), 24)
    }

    func testMean() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        XCTAssertEqual(a.mean(), 3)
    }

    func testVariance() {
        let a = NDArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        XCTAssertEqual(a.variance(), 4, accuracy: epsilon)
    }

    func testStd() {
        let a = NDArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        XCTAssertEqual(a.std(), 2, accuracy: epsilon)
    }

    func testMin() {
        let a = NDArray([3.0, 1.0, 4.0, 1.0, 5.0])
        XCTAssertEqual(a.min(), 1)
    }

    func testMax() {
        let a = NDArray([3.0, 1.0, 4.0, 1.0, 5.0])
        XCTAssertEqual(a.max(), 5)
    }

    func testArgmin() {
        let a = NDArray([3.0, 1.0, 4.0, 1.0, 5.0])
        XCTAssertEqual(a.argmin(), 1)
    }

    func testArgmax() {
        let a = NDArray([3.0, 1.0, 4.0, 5.0, 2.0])
        XCTAssertEqual(a.argmax(), 3)
    }

    func testCumsum() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = a.cumsum()
        XCTAssertEqual(b.real, [1, 3, 6, 10])
    }

    func testCumprod() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = a.cumprod()
        XCTAssertEqual(b.real, [1, 2, 6, 24])
    }

    func testDiff() {
        let a = NDArray([1.0, 3.0, 6.0, 10.0])
        let b = a.diff()
        XCTAssertEqual(b.real, [2, 3, 4])
    }

    func testMedian() {
        let a = NDArray([1.0, 3.0, 2.0])
        XCTAssertEqual(a.median(), 2)

        let b = NDArray([1.0, 2.0, 3.0, 4.0])
        XCTAssertEqual(b.median(), 2.5)
    }

    func testPercentile() {
        let a = NDArray.linspace(start: 0, stop: 100, num: 101)
        XCTAssertEqual(a.percentile(50), 50, accuracy: epsilon)
        XCTAssertEqual(a.percentile(25), 25, accuracy: epsilon)
        XCTAssertEqual(a.percentile(75), 75, accuracy: epsilon)
    }

    func testAll() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([1.0, 0.0, 3.0])
        XCTAssertTrue(a.all())
        XCTAssertFalse(b.all())
    }

    func testAny() {
        let a = NDArray([0.0, 0.0, 1.0])
        let b = NDArray([0.0, 0.0, 0.0])
        XCTAssertTrue(a.any())
        XCTAssertFalse(b.any())
    }

    // MARK: - Shape Manipulation Tests

    func testReshape() {
        let a = NDArray.arange(start: 0, stop: 12, step: 1)
        let b = a.reshape([3, 4])
        XCTAssertEqual(b.shape, [3, 4])
        XCTAssertEqual(b.real, a.real)
    }

    func testFlatten() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let b = a.flatten()
        XCTAssertEqual(b.shape, [6])
        XCTAssertEqual(b.real, [1, 2, 3, 4, 5, 6])
    }

    func testSqueeze() {
        let a = NDArray.zeros([1, 3, 1, 4])
        let b = a.squeeze()
        XCTAssertEqual(b.shape, [3, 4])
    }

    func testSqueezeAxis() {
        let a = NDArray.zeros([1, 3, 1, 4])
        let b = a.squeeze(axis: 0)
        XCTAssertEqual(b.shape, [3, 1, 4])
    }

    func testExpandDims() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = a.expandDims(axis: 0)
        let c = a.expandDims(axis: 1)
        XCTAssertEqual(b.shape, [1, 3])
        XCTAssertEqual(c.shape, [3, 1])
    }

    func testTranspose() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let b = a.transpose()
        XCTAssertEqual(b.shape, [3, 2])
        XCTAssertEqual(b.real[0], 1)  // [0,0]
        XCTAssertEqual(b.real[1], 4)  // [0,1]
        XCTAssertEqual(b.real[2], 2)  // [1,0]
    }

    func testComplexTranspose() {
        // Create 2x3 complex array
        let a = NDArray.complexArray(
            shape: [2, 3],
            real: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            imag: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        )
        let b = a.transpose()
        XCTAssertEqual(b.shape, [3, 2])
        XCTAssertTrue(b.isComplex)
        // Original: [[1+10i, 2+20i, 3+30i], [4+40i, 5+50i, 6+60i]]
        // Transposed: [[1+10i, 4+40i], [2+20i, 5+50i], [3+30i, 6+60i]]
        XCTAssertEqual(b.real[0], 1)   // [0,0] real
        XCTAssertEqual(b.imag![0], 10) // [0,0] imag
        XCTAssertEqual(b.real[1], 4)   // [0,1] real
        XCTAssertEqual(b.imag![1], 40) // [0,1] imag
        XCTAssertEqual(b.real[2], 2)   // [1,0] real
        XCTAssertEqual(b.imag![2], 20) // [1,0] imag
    }

    func testComplexConcatenate() {
        let a = NDArray.complexArray(shape: [2], real: [1.0, 2.0], imag: [10.0, 20.0])
        let b = NDArray.complexArray(shape: [2], real: [3.0, 4.0], imag: [30.0, 40.0])
        let c = NDArray.concatenate([a, b])
        XCTAssertEqual(c.shape, [4])
        XCTAssertTrue(c.isComplex)
        XCTAssertEqual(c.real, [1, 2, 3, 4])
        XCTAssertEqual(c.imag!, [10, 20, 30, 40])
    }

    func testComplexSplit() {
        let a = NDArray.complexArray(shape: [4], real: [1.0, 2.0, 3.0, 4.0], imag: [10.0, 20.0, 30.0, 40.0])
        let parts = a.split(indices: [2])
        XCTAssertEqual(parts.count, 2)
        XCTAssertTrue(parts[0].isComplex)
        XCTAssertTrue(parts[1].isComplex)
        XCTAssertEqual(parts[0].real, [1, 2])
        XCTAssertEqual(parts[0].imag!, [10, 20])
        XCTAssertEqual(parts[1].real, [3, 4])
        XCTAssertEqual(parts[1].imag!, [30, 40])
    }

    func testComplexFlip() {
        let a = NDArray.complexArray(shape: [3], real: [1.0, 2.0, 3.0], imag: [10.0, 20.0, 30.0])
        let b = a.flip()
        XCTAssertTrue(b.isComplex)
        XCTAssertEqual(b.real, [3, 2, 1])
        XCTAssertEqual(b.imag!, [30, 20, 10])
    }

    func testComplexRoll() {
        let a = NDArray.complexArray(shape: [4], real: [1.0, 2.0, 3.0, 4.0], imag: [10.0, 20.0, 30.0, 40.0])
        let b = a.roll(1)
        XCTAssertTrue(b.isComplex)
        XCTAssertEqual(b.real, [4, 1, 2, 3])
        XCTAssertEqual(b.imag!, [40, 10, 20, 30])
    }

    func testComplexRepeat() {
        let a = NDArray.complexArray(shape: [2], real: [1.0, 2.0], imag: [10.0, 20.0])
        let b = a.repeat(2)
        XCTAssertTrue(b.isComplex)
        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.real, [1, 1, 2, 2])
        XCTAssertEqual(b.imag!, [10, 10, 20, 20])
    }

    func testComplexMathFunctions() {
        // Test complex square: (3+4i)² = 9-16+24i = -7+24i
        let a = NDArray.complexArray(shape: [1], real: [3.0], imag: [4.0])
        let sq = a.square()
        XCTAssertTrue(sq.isComplex)
        XCTAssertEqual(sq.real[0], -7, accuracy: epsilon)
        XCTAssertEqual(sq.imag![0], 24, accuracy: epsilon)

        // Test complex sqrt: sqrt(-1) = i
        let minusOne = NDArray.complexArray(shape: [1], real: [-1.0], imag: [0.0])
        let sqrtMinusOne = minusOne.sqrt()
        XCTAssertTrue(sqrtMinusOne.isComplex)
        XCTAssertEqual(sqrtMinusOne.real[0], 0, accuracy: epsilon)
        XCTAssertEqual(sqrtMinusOne.imag![0], 1, accuracy: epsilon)

        // Test complex sign: sign(3+4i) = (3+4i)/5 = 0.6+0.8i
        let sign = a.sign()
        XCTAssertTrue(sign.isComplex)
        XCTAssertEqual(sign.real[0], 0.6, accuracy: epsilon)
        XCTAssertEqual(sign.imag![0], 0.8, accuracy: epsilon)
    }

    func testConcatenate() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([4.0, 5.0, 6.0])
        let c = NDArray.concatenate([a, b])
        XCTAssertEqual(c.shape, [6])
        XCTAssertEqual(c.real, [1, 2, 3, 4, 5, 6])
    }

    func testConcatenateAxis() {
        let a = NDArray([[1.0, 2.0], [3.0, 4.0]])
        let b = NDArray([[5.0, 6.0], [7.0, 8.0]])

        let c0 = NDArray.concatenate([a, b], axis: 0)
        XCTAssertEqual(c0.shape, [4, 2])

        let c1 = NDArray.concatenate([a, b], axis: 1)
        XCTAssertEqual(c1.shape, [2, 4])
    }

    func testStack() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([4.0, 5.0, 6.0])
        let c = NDArray.stack([a, b])
        XCTAssertEqual(c.shape, [2, 3])
    }

    func testSplit() {
        let a = NDArray.arange(start: 0, stop: 9, step: 1)
        let parts = a.split(indices: [3, 6])
        XCTAssertEqual(parts.count, 3)
        XCTAssertEqual(parts[0].real, [0, 1, 2])
        XCTAssertEqual(parts[1].real, [3, 4, 5])
        XCTAssertEqual(parts[2].real, [6, 7, 8])
    }

    func testTile() {
        let a = NDArray([1.0, 2.0])
        let b = a.tile([3])
        XCTAssertEqual(b.real, [1, 2, 1, 2, 1, 2])
    }

    func testRepeat() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = a.repeat(2)
        XCTAssertEqual(b.real, [1, 1, 2, 2, 3, 3])
    }

    func testFlip() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = a.flip()
        XCTAssertEqual(b.real, [4, 3, 2, 1])
    }

    func testRoll() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = a.roll(1)
        XCTAssertEqual(b.real, [4, 1, 2, 3])
    }

    func testCopy() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = a.copy()
        XCTAssertEqual(b.real, a.real)
        XCTAssertEqual(b.shape, a.shape)
    }

    // MARK: - Broadcasting Tests

    func testBroadcastAdd() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  // 2x3
        let b = NDArray([10.0, 20.0, 30.0])  // 3
        let c = a + b
        XCTAssertEqual(c.shape, [2, 3])
        XCTAssertEqual(c.real, [11, 22, 33, 14, 25, 36])
    }

    // MARK: - Complex Number Tests

    func testComplexCreation() {
        let c = NDArray.complexArray(
            shape: [2],
            real: [1.0, 2.0],
            imag: [3.0, 4.0]
        )
        XCTAssertTrue(c.isComplex)
        XCTAssertEqual(c.real, [1, 2])
        XCTAssertEqual(c.imag, [3, 4])
    }

    func testComplexAbs() {
        let c = NDArray.complexArray(
            shape: [2],
            real: [3.0, 0.0],
            imag: [4.0, 5.0]
        )
        let mag = c.abs()
        XCTAssertEqual(mag.real[0], 5, accuracy: epsilon)
        XCTAssertEqual(mag.real[1], 5, accuracy: epsilon)
    }

    func testComplexConjugate() {
        let c = NDArray.complexArray(
            shape: [2],
            real: [1.0, 2.0],
            imag: [3.0, 4.0]
        )
        let conj = c.conjugate()
        XCTAssertEqual(conj.real, [1, 2])
        XCTAssertEqual(conj.imag, [-3, -4])
    }

    func testComplexAdd() {
        let a = NDArray.complexArray(shape: [2], real: [1.0, 2.0], imag: [3.0, 4.0])
        let b = NDArray.complexArray(shape: [2], real: [5.0, 6.0], imag: [7.0, 8.0])
        let c = a + b
        XCTAssertEqual(c.real, [6, 8])
        XCTAssertEqual(c.imag, [10, 12])
    }

    func testComplexMultiply() {
        // (1+2i) * (3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let a = NDArray.complexArray(shape: [1], real: [1.0], imag: [2.0])
        let b = NDArray.complexArray(shape: [1], real: [3.0], imag: [4.0])
        let c = a * b
        XCTAssertEqual(c.real[0], -5, accuracy: epsilon)
        XCTAssertEqual(c.imag![0], 10, accuracy: epsilon)
    }

    // MARK: - Free Function Tests

    func testFreeFunctions() {
        let a = NDArray([1.0, 2.0, 3.0])

        XCTAssertEqual(ArraySwift.sum(a), 6)
        XCTAssertEqual(ArraySwift.mean(a), 2)
        XCTAssertEqual(ArraySwift.min(a), 1)
        XCTAssertEqual(ArraySwift.max(a), 3)

        let b = ArraySwift.sqrt(NDArray([4.0, 9.0, 16.0]))
        XCTAssertEqual(b.real, [2, 3, 4])
    }

    // MARK: - Convenience Init Tests

    func testInit1D() {
        let a = NDArray([1.0, 2.0, 3.0])
        XCTAssertEqual(a.shape, [3])
        XCTAssertEqual(a.real, [1, 2, 3])
    }

    func testInit2D() {
        let a = NDArray([[1.0, 2.0], [3.0, 4.0]])
        XCTAssertEqual(a.shape, [2, 2])
        XCTAssertEqual(a.real, [1, 2, 3, 4])
    }

    func testInit3D() {
        let a = NDArray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        XCTAssertEqual(a.shape, [2, 2, 2])
        XCTAssertEqual(a.size, 8)
    }

    // MARK: - Complex Reduction Tests

    func testComplexReductions() {
        // Test complex sum
        let a = NDArray.complexArray(shape: [3], real: [1.0, 2.0, 3.0], imag: [4.0, 5.0, 6.0])
        let sumResult = a.complexSum()
        XCTAssertEqual(sumResult.real, 6.0, accuracy: epsilon)
        XCTAssertEqual(sumResult.imag, 15.0, accuracy: epsilon)

        // Test complex prod: (1+4i)(2+5i)(3+6i)
        // (1+4i)(2+5i) = 2 + 5i + 8i + 20i² = 2 + 13i - 20 = -18 + 13i
        // (-18+13i)(3+6i) = -54 - 108i + 39i + 78i² = -54 - 69i - 78 = -132 - 69i
        let prodResult = a.complexProd()
        XCTAssertEqual(prodResult.real, -132.0, accuracy: epsilon)
        XCTAssertEqual(prodResult.imag, -69.0, accuracy: epsilon)

        // Test complex mean
        let meanResult = a.complexMean()
        XCTAssertEqual(meanResult.real, 2.0, accuracy: epsilon)
        XCTAssertEqual(meanResult.imag, 5.0, accuracy: epsilon)

        // Test complex variance: mean of |z - mean|²
        // Values: (1+4i), (2+5i), (3+6i), mean = (2+5i)
        // |1+4i - 2-5i|² = |-1-i|² = 1 + 1 = 2
        // |2+5i - 2-5i|² = |0|² = 0
        // |3+6i - 2-5i|² = |1+i|² = 1 + 1 = 2
        // variance = (2 + 0 + 2) / 3 = 4/3
        let variance = a.variance()
        XCTAssertEqual(variance, 4.0/3.0, accuracy: epsilon)

        // Test complex cumsum
        let cumsum = a.cumsum()
        XCTAssertTrue(cumsum.isComplex)
        XCTAssertEqual(cumsum.real, [1, 3, 6])
        XCTAssertEqual(cumsum.imag, [4, 9, 15])

        // Test complex cumprod
        let cumprod = a.cumprod()
        XCTAssertTrue(cumprod.isComplex)
        XCTAssertEqual(cumprod.real[0], 1.0, accuracy: epsilon)  // (1+4i)
        XCTAssertEqual(cumprod.imag![0], 4.0, accuracy: epsilon)
        XCTAssertEqual(cumprod.real[1], -18.0, accuracy: epsilon)  // (1+4i)(2+5i)
        XCTAssertEqual(cumprod.imag![1], 13.0, accuracy: epsilon)
        XCTAssertEqual(cumprod.real[2], -132.0, accuracy: epsilon)  // final product
        XCTAssertEqual(cumprod.imag![2], -69.0, accuracy: epsilon)

        // Test complex diff
        let diff = a.diff()
        XCTAssertTrue(diff.isComplex)
        XCTAssertEqual(diff.real, [1, 1])  // 2-1, 3-2
        XCTAssertEqual(diff.imag, [1, 1])  // 5-4, 6-5
    }

    func testComplexSumAlongAxis() {
        // 2x2 complex array
        let a = NDArray.complexArray(
            shape: [2, 2],
            real: [1.0, 2.0, 3.0, 4.0],
            imag: [5.0, 6.0, 7.0, 8.0]
        )

        // Sum along axis 0 (columns)
        let sum0 = a.sum(axis: 0)
        XCTAssertTrue(sum0.isComplex)
        XCTAssertEqual(sum0.shape, [2])
        XCTAssertEqual(sum0.real, [4, 6])  // 1+3, 2+4
        XCTAssertEqual(sum0.imag, [12, 14])  // 5+7, 6+8

        // Sum along axis 1 (rows)
        let sum1 = a.sum(axis: 1)
        XCTAssertTrue(sum1.isComplex)
        XCTAssertEqual(sum1.shape, [2])
        XCTAssertEqual(sum1.real, [3, 7])  // 1+2, 3+4
        XCTAssertEqual(sum1.imag, [11, 15])  // 5+6, 7+8
    }

    // MARK: - Subscript Tests

    func testSubscript1D() {
        var a = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])

        // Get
        XCTAssertEqual(a[0], 1.0)
        XCTAssertEqual(a[2], 3.0)
        XCTAssertEqual(a[4], 5.0)

        // Set
        a[1] = 10.0
        XCTAssertEqual(a[1], 10.0)
    }

    func testSubscript2D() {
        var a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  // 2x3

        // Get
        XCTAssertEqual(a[0, 0], 1.0)
        XCTAssertEqual(a[0, 2], 3.0)
        XCTAssertEqual(a[1, 1], 5.0)

        // Set
        a[1, 0] = 100.0
        XCTAssertEqual(a[1, 0], 100.0)
    }

    func testSubscript3D() {
        let a = NDArray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  // 2x2x2

        // Get
        XCTAssertEqual(a[0, 0, 0], 1.0)
        XCTAssertEqual(a[0, 1, 1], 4.0)
        XCTAssertEqual(a[1, 1, 0], 7.0)
    }

    func testSubscriptArrayIndices() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  // 2x3

        XCTAssertEqual(a[[0, 0]], 1.0)
        XCTAssertEqual(a[[0, 2]], 3.0)
        XCTAssertEqual(a[[1, 1]], 5.0)
    }

    func testSubscriptRowColumn() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  // 2x3

        // Row access
        let row0 = a[row: 0]
        XCTAssertEqual(row0.shape, [3])
        XCTAssertEqual(row0.real, [1, 2, 3])

        let row1 = a[row: 1]
        XCTAssertEqual(row1.real, [4, 5, 6])

        // Column access
        let col0 = a[col: 0]
        XCTAssertEqual(col0.shape, [2])
        XCTAssertEqual(col0.real, [1, 4])

        let col2 = a[col: 2]
        XCTAssertEqual(col2.real, [3, 6])
    }

    func testSubscriptRange1D() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])

        let slice = a[1..<4]
        XCTAssertEqual(slice.shape, [3])
        XCTAssertEqual(slice.real, [2, 3, 4])
    }

    func testSubscriptRange2D() {
        let a = NDArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  // 3x3

        let slice = a[0..<2, 1..<3]
        XCTAssertEqual(slice.shape, [2, 2])
        XCTAssertEqual(slice.real, [2, 3, 5, 6])
    }

    func testSubscriptComplex() {
        var a = NDArray.complexArray(shape: [3], real: [1.0, 2.0, 3.0], imag: [4.0, 5.0, 6.0])

        // Get complex element
        let elem = a[complex: 1]
        XCTAssertEqual(elem.real, 2.0)
        XCTAssertEqual(elem.imag, 5.0)

        // Set complex element
        a[complex: 0] = (real: 10.0, imag: 20.0)
        let updated = a[complex: 0]
        XCTAssertEqual(updated.real, 10.0)
        XCTAssertEqual(updated.imag, 20.0)
    }

    func testSubscriptComplexRowColumn() {
        let a = NDArray.complexArray(
            shape: [2, 2],
            real: [1.0, 2.0, 3.0, 4.0],
            imag: [5.0, 6.0, 7.0, 8.0]
        )

        // Row access
        let row0 = a[row: 0]
        XCTAssertTrue(row0.isComplex)
        XCTAssertEqual(row0.real, [1, 2])
        XCTAssertEqual(row0.imag, [5, 6])

        // Column access
        let col1 = a[col: 1]
        XCTAssertTrue(col1.isComplex)
        XCTAssertEqual(col1.real, [2, 4])
        XCTAssertEqual(col1.imag, [6, 8])
    }
}
