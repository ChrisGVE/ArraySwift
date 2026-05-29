//
//  NDArraySetOpsTests.swift
//  ArraySwiftTests
//
//  Tests for 1-D set operations: intersect1d, union1d, setdiff1d, setxor1d, in1d.
//

import XCTest
@testable import ArraySwift

final class NDArraySetOpsTests: XCTestCase {

    // MARK: - intersect1d

    func testIntersect1dFloat64() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = NDArray([2.0, 4.0, 6.0])
        let result = NDArray.intersect1d(a, b)
        XCTAssertEqual(result.real, [2.0, 4.0])
    }

    func testIntersect1dDisjoint() {
        let a = NDArray([1.0, 2.0])
        let b = NDArray([3.0, 4.0])
        let result = NDArray.intersect1d(a, b)
        XCTAssertEqual(result.size, 0)
    }

    func testIntersect1dDuplicates() {
        let a = NDArray([1.0, 1.0, 2.0, 3.0])
        let b = NDArray([1.0, 2.0, 2.0])
        let result = NDArray.intersect1d(a, b)
        XCTAssertEqual(result.real, [1.0, 2.0])  // sorted unique
    }

    func testIntersect1dInt64() {
        let a = NDArray(shape: [4], int64Data: [1, 2, 3, 4])
        let b = NDArray(shape: [3], int64Data: [2, 4, 6])
        let result = NDArray.intersect1d(a, b)
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [2, 4])
    }

    // MARK: - union1d

    func testUnion1dFloat64() {
        let a = NDArray([1.0, 3.0, 5.0])
        let b = NDArray([2.0, 3.0, 4.0])
        let result = NDArray.union1d(a, b)
        XCTAssertEqual(result.real, [1.0, 2.0, 3.0, 4.0, 5.0])
    }

    func testUnion1dDuplicates() {
        let a = NDArray([1.0, 2.0, 2.0])
        let b = NDArray([2.0, 3.0])
        let result = NDArray.union1d(a, b)
        XCTAssertEqual(result.real, [1.0, 2.0, 3.0])
    }

    func testUnion1dInt64() {
        let a = NDArray(shape: [3], int64Data: [1, 3, 5])
        let b = NDArray(shape: [3], int64Data: [2, 3, 4])
        let result = NDArray.union1d(a, b)
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [1, 2, 3, 4, 5])
    }

    // MARK: - setdiff1d

    func testSetdiff1dFloat64() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0])
        let b = NDArray([2.0, 4.0])
        let result = NDArray.setdiff1d(a, b)
        XCTAssertEqual(result.real, [1.0, 3.0])
    }

    func testSetdiff1dEmpty() {
        let a = NDArray([1.0, 2.0])
        let b = NDArray([1.0, 2.0, 3.0])
        let result = NDArray.setdiff1d(a, b)
        XCTAssertEqual(result.size, 0)
    }

    func testSetdiff1dInt64() {
        let a = NDArray(shape: [4], int64Data: [1, 2, 3, 4])
        let b = NDArray(shape: [2], int64Data: [2, 4])
        let result = NDArray.setdiff1d(a, b)
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [1, 3])
    }

    // MARK: - setxor1d

    func testSetxor1dFloat64() {
        let a = NDArray([1.0, 2.0, 3.0])
        let b = NDArray([2.0, 3.0, 4.0])
        let result = NDArray.setxor1d(a, b)
        XCTAssertEqual(result.real, [1.0, 4.0])
    }

    func testSetxor1dInt64() {
        let a = NDArray(shape: [3], int64Data: [1, 2, 3])
        let b = NDArray(shape: [3], int64Data: [2, 3, 4])
        let result = NDArray.setxor1d(a, b)
        XCTAssertEqual(result.dtype, .int64)
        XCTAssertEqual(result.int64Data, [1, 4])
    }

    // MARK: - in1d

    func testIn1dFloat64() {
        let a = NDArray([1.0, 2.0, 3.0, 4.0, 5.0])
        let b = NDArray([2.0, 4.0])
        let result = NDArray.in1d(a, b)
        XCTAssertEqual(result.dtype, .bool)
        XCTAssertEqual(result.boolData, [0, 1, 0, 1, 0])
    }

    func testIn1dAllTrue() {
        let a = NDArray([1.0, 2.0])
        let b = NDArray([1.0, 2.0, 3.0])
        let result = NDArray.in1d(a, b)
        XCTAssertEqual(result.boolData, [1, 1])
    }

    func testIn1dAllFalse() {
        let a = NDArray([5.0, 6.0])
        let b = NDArray([1.0, 2.0])
        let result = NDArray.in1d(a, b)
        XCTAssertEqual(result.boolData, [0, 0])
    }

    func testIn1dInt64() {
        let a = NDArray(shape: [4], int64Data: [10, 20, 30, 40])
        let b = NDArray(shape: [2], int64Data: [20, 40])
        let result = NDArray.in1d(a, b)
        XCTAssertEqual(result.dtype, .bool)
        XCTAssertEqual(result.boolData, [0, 1, 0, 1])
    }

    // MARK: - Edge: empty inputs

    func testSetOpsEmptyLeft() {
        let a = NDArray([Double]())  // requires a workaround – use zeros(0)
        let b = NDArray([1.0, 2.0])
        let inter = NDArray.intersect1d(a, b)
        XCTAssertEqual(inter.size, 0)
        let union = NDArray.union1d(a, b)
        XCTAssertEqual(union.real, [1.0, 2.0])
    }

    func testIn1dEmptyLeft() {
        let a = NDArray(shape: [0], storage: .float64([]))
        let b = NDArray([1.0, 2.0])
        let result = NDArray.in1d(a, b)
        XCTAssertEqual(result.size, 0)
    }
}
