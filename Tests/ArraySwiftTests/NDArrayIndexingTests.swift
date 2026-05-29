//
//  NDArrayIndexingTests.swift
//  ArraySwiftTests
//
//  Tests for negative-index normalisation, boolean indexing, and fancy indexing.
//

import XCTest
@testable import ArraySwift

final class NDArrayIndexingTests: XCTestCase {

    // MARK: - Negative-index normalisation (element subscript)

    func testNegativeIndexSingleElement() {
        let a = NDArray([10.0, 20.0, 30.0, 40.0, 50.0])
        XCTAssertEqual(a[-1], 50.0)
        XCTAssertEqual(a[-2], 40.0)
        XCTAssertEqual(a[-5], 10.0)
    }

    func testNegativeIndexSet() {
        var a = NDArray([1.0, 2.0, 3.0])
        a[-1] = 99.0
        XCTAssertEqual(a[2], 99.0)
    }

    func testNegativeIndex2D() {
        let m = NDArray(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(m[-1, -1], 6.0)
        XCTAssertEqual(m[-2, -1], 3.0)
        XCTAssertEqual(m[-1, 0], 4.0)
    }

    func testNormaliseIndexHelper() {
        let a = NDArray([0.0, 1.0, 2.0, 3.0])
        XCTAssertEqual(a.normalizeIndex(-1, size: 4), 3)
        XCTAssertEqual(a.normalizeIndex(-4, size: 4), 0)
        XCTAssertEqual(a.normalizeIndex(0, size: 4), 0)
        XCTAssertEqual(a.normalizeIndex(3, size: 4), 3)
    }

    // MARK: - Boolean indexing (getter via subscript(mask:))

    func testBooleanIndexingBasic() {
        let a = NDArray([10.0, 20.0, 30.0, 40.0, 50.0])
        let mask = NDArray(shape: [5], boolData: [true, false, true, false, true])
        let result = a[mask: mask]
        XCTAssertEqual(result.shape, [3])
        XCTAssertEqual(result.real, [10.0, 30.0, 50.0])
    }

    func testBooleanIndexingAllFalse() {
        let a = NDArray([1.0, 2.0, 3.0])
        let mask = NDArray(shape: [3], boolData: [false, false, false])
        let result = a[mask: mask]
        XCTAssertEqual(result.shape, [0])
        XCTAssertEqual(result.size, 0)
    }

    func testBooleanIndexingAllTrue() {
        let a = NDArray([7.0, 8.0, 9.0])
        let mask = NDArray(shape: [3], boolData: [true, true, true])
        let result = a[mask: mask]
        XCTAssertEqual(result.real, [7.0, 8.0, 9.0])
    }

    func testBooleanIndexingInt64() {
        let a = NDArray(shape: [4], int64Data: [10, 20, 30, 40])
        let mask = NDArray(shape: [4], boolData: [false, true, false, true])
        let result = a[mask: mask]
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [20, 40])
    }

    func testBooleanIndexingComplex() {
        let a = NDArray.complexArray(shape: [3], real: [1, 2, 3], imag: [4, 5, 6])
        let mask = NDArray(shape: [3], boolData: [true, false, true])
        let result = a[mask: mask]
        XCTAssertEqual(result.dtype, .complex128)
        XCTAssertEqual(result.real, [1.0, 3.0])
        XCTAssertEqual(result.imag, [4.0, 6.0])
    }

    // MARK: - Boolean indexing (setter via maskSet / booleanSetInt64)

    func testBooleanIndexingSetterFloat64() {
        var a = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        let mask = NDArray(shape: [5], boolData: [false, true, false, true, false])
        a.maskSet(mask, value: 99.0)
        XCTAssertEqual(a.real, [1.0, 99.0, 3.0, 99.0, 5.0])
    }

    func testBooleanIndexingSetterInt64() {
        var a = NDArray(shape: [4], int64Data: [1, 2, 3, 4])
        let mask = NDArray(shape: [4], boolData: [true, false, false, true])
        a.booleanSetInt64(mask, to: 0)
        XCTAssertEqual(a.int64Data, [0, 2, 3, 0])
    }

    // MARK: - Fancy indexing ([Int])

    func testFancyIndexingBasic() {
        let a = NDArray([10.0, 20.0, 30.0, 40.0, 50.0])
        let result = a[indices: [0, 2, 4]]
        XCTAssertEqual(result.real, [10.0, 30.0, 50.0])
    }

    func testFancyIndexingNegative() {
        let a = NDArray([10.0, 20.0, 30.0, 40.0, 50.0])
        let result = a[indices: [-1, -3, 0]]
        XCTAssertEqual(result.real, [50.0, 30.0, 10.0])
    }

    func testFancyIndexingRepeat() {
        let a = NDArray([1.0, 2.0, 3.0])
        let result = a[indices: [0, 0, 2, 2]]
        XCTAssertEqual(result.real, [1.0, 1.0, 3.0, 3.0])
    }

    func testFancyIndexingInt64() {
        let a = NDArray(shape: [4], int64Data: [10, 20, 30, 40])
        let result = a[indices: [3, 1]]
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [40, 20])
    }

    func testFancyIndexingComplex() {
        let a = NDArray.complexArray(shape: [3], real: [1, 2, 3], imag: [4, 5, 6])
        let result = a[indices: [2, 0]]
        XCTAssertEqual(result.dtype, .complex128)
        XCTAssertEqual(result.real, [3.0, 1.0])
        XCTAssertEqual(result.imag, [6.0, 4.0])
    }

    func testFancyIndexingEmptyResult() {
        let a = NDArray([1.0, 2.0, 3.0])
        let result = a[indices: []]
        XCTAssertEqual(result.shape, [0])
    }

    // MARK: - Fancy indexing with NDArray of int64

    func testFancyIndexingNDArray() {
        let a = NDArray([10.0, 20.0, 30.0, 40.0])
        let idx = NDArray(shape: [3], int64Data: [3, 0, 2])
        let result = a[ndIndices: idx]
        XCTAssertEqual(result.real, [40.0, 10.0, 30.0])
    }
}
