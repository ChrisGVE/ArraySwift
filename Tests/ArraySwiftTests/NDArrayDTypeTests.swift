//
//  NDArrayDTypeTests.swift
//  ArraySwiftTests
//
//  Tests for multi-dtype support: int64, bool, date
//

import XCTest
@testable import ArraySwift

final class NDArrayDTypeTests: XCTestCase {

    // MARK: - ArrayDType Tests

    func testDTypeEnum() {
        XCTAssertEqual(ArrayDType.float64.rawValue, "float64")
        XCTAssertEqual(ArrayDType.complex128.rawValue, "complex128")
        XCTAssertEqual(ArrayDType.int64.rawValue, "int64")
        XCTAssertEqual(ArrayDType.bool.rawValue, "bool")
        XCTAssertEqual(ArrayDType.date.rawValue, "date")
    }

    func testDTypeBytesPerElement() {
        XCTAssertEqual(ArrayDType.float64.bytesPerElement, 8)
        XCTAssertEqual(ArrayDType.complex128.bytesPerElement, 16)
        XCTAssertEqual(ArrayDType.int64.bytesPerElement, 8)
        XCTAssertEqual(ArrayDType.bool.bytesPerElement, 1)
        XCTAssertEqual(ArrayDType.date.bytesPerElement, 8)
    }

    func testDTypeIsComplex() {
        XCTAssertFalse(ArrayDType.float64.isComplex)
        XCTAssertTrue(ArrayDType.complex128.isComplex)
        XCTAssertFalse(ArrayDType.int64.isComplex)
        XCTAssertFalse(ArrayDType.bool.isComplex)
        XCTAssertFalse(ArrayDType.date.isComplex)
    }

    func testDTypeIsInteger() {
        XCTAssertFalse(ArrayDType.float64.isInteger)
        XCTAssertFalse(ArrayDType.complex128.isInteger)
        XCTAssertTrue(ArrayDType.int64.isInteger)
        XCTAssertTrue(ArrayDType.bool.isInteger)
        XCTAssertFalse(ArrayDType.date.isInteger)
    }

    func testDTypeInit() {
        XCTAssertEqual(ArrayDType(from: "int64"), .int64)
        XCTAssertEqual(ArrayDType(from: "bool"), .bool)
        XCTAssertEqual(ArrayDType(from: "date"), .date)
        XCTAssertEqual(ArrayDType(from: "float64"), .float64)
        XCTAssertEqual(ArrayDType(from: "complex128"), .complex128)
        XCTAssertEqual(ArrayDType(from: nil), .float64)
        XCTAssertEqual(ArrayDType(from: "unknown"), .float64)
    }

    // MARK: - Dtype Promotion Tests

    func testPromotionTable() {
        XCTAssertEqual(ArrayDType.promote(.bool, .bool), .bool)
        XCTAssertEqual(ArrayDType.promote(.bool, .int64), .int64)
        XCTAssertEqual(ArrayDType.promote(.bool, .float64), .float64)
        XCTAssertEqual(ArrayDType.promote(.bool, .complex128), .complex128)
        XCTAssertEqual(ArrayDType.promote(.int64, .int64), .int64)
        XCTAssertEqual(ArrayDType.promote(.int64, .float64), .float64)
        XCTAssertEqual(ArrayDType.promote(.int64, .complex128), .complex128)
        XCTAssertEqual(ArrayDType.promote(.float64, .float64), .float64)
        XCTAssertEqual(ArrayDType.promote(.float64, .complex128), .complex128)
        XCTAssertEqual(ArrayDType.promote(.complex128, .complex128), .complex128)
    }

    func testPromotionSymmetry() {
        let types: [ArrayDType] = [.bool, .int64, .float64, .complex128]
        for a in types {
            for b in types {
                XCTAssertEqual(ArrayDType.promote(a, b), ArrayDType.promote(b, a),
                               "Promotion not symmetric for \(a) and \(b)")
            }
        }
    }

    func testDatePromotionReturnsNil() {
        XCTAssertNil(ArrayDType.promoteOrNil(.date, .int64))
        XCTAssertNil(ArrayDType.promoteOrNil(.date, .float64))
        XCTAssertNil(ArrayDType.promoteOrNil(.date, .complex128))
        XCTAssertNil(ArrayDType.promoteOrNil(.date, .bool))
        // date+date is special (subtraction yields float64 duration), handled separately
    }

    // MARK: - Int64 NDArray Tests

    func testInt64Creation() {
        let arr = NDArray(shape: [3], int64Data: [1, 2, 3])
        XCTAssertEqual(arr.dtype, .int64)
        XCTAssertEqual(arr.shape, [3])
        XCTAssertEqual(arr.size, 3)
        XCTAssertEqual(arr.int64Data, [1, 2, 3])
    }

    func testInt64RealAccessor() {
        let arr = NDArray(shape: [3], int64Data: [10, 20, 30])
        XCTAssertEqual(arr.real, [10.0, 20.0, 30.0])
    }

    func testInt64ZerosFactory() {
        let arr = NDArray.zeros([2, 3], dtype: .int64)
        XCTAssertEqual(arr.dtype, .int64)
        XCTAssertEqual(arr.int64Data, [Int64](repeating: 0, count: 6))
    }

    func testInt64OnesFactory() {
        let arr = NDArray.ones([4], dtype: .int64)
        XCTAssertEqual(arr.dtype, .int64)
        XCTAssertEqual(arr.int64Data, [1, 1, 1, 1])
    }

    func testInt64Arithmetic() {
        let a = NDArray(shape: [3], int64Data: [1, 2, 3])
        let b = NDArray(shape: [3], int64Data: [4, 5, 6])
        let c = a + b
        XCTAssertEqual(c.dtype, .int64)
        XCTAssertEqual(c.int64Data, [5, 7, 9])
    }

    func testInt64Subtract() {
        let a = NDArray(shape: [3], int64Data: [10, 20, 30])
        let b = NDArray(shape: [3], int64Data: [1, 2, 3])
        let c = a - b
        XCTAssertEqual(c.dtype, .int64)
        XCTAssertEqual(c.int64Data, [9, 18, 27])
    }

    func testInt64Multiply() {
        let a = NDArray(shape: [3], int64Data: [2, 3, 4])
        let b = NDArray(shape: [3], int64Data: [3, 4, 5])
        let c = a * b
        XCTAssertEqual(c.dtype, .int64)
        XCTAssertEqual(c.int64Data, [6, 12, 20])
    }

    func testInt64PromotedToFloat64() {
        let a = NDArray(shape: [3], int64Data: [1, 2, 3])
        let b = NDArray(shape: [3], data: [1.5, 2.5, 3.5])
        let c = a + b
        XCTAssertEqual(c.dtype, .float64)
        XCTAssertEqual(c.real, [2.5, 4.5, 6.5])
    }

    func testInt64PromotedToComplex() {
        let a = NDArray(shape: [2], int64Data: [1, 2])
        let b = NDArray(shape: [2], dtype: .complex128, real: [1.0, 2.0], imag: [3.0, 4.0])
        let c = a + b
        XCTAssertEqual(c.dtype, .complex128)
        XCTAssertEqual(c.real, [2.0, 4.0])
        XCTAssertEqual(c.imag, [3.0, 4.0])
    }

    func testArgmaxArrayReturnsInt64() {
        let arr = NDArray(shape: [5], data: [3.0, 1.0, 4.0, 1.0, 5.0])
        let idx = arr.argmaxArray()
        XCTAssertEqual(idx.dtype, .int64)
        XCTAssertEqual(idx.int64Data, [4])
    }

    func testArgminArrayReturnsInt64() {
        let arr = NDArray(shape: [5], data: [3.0, 1.0, 4.0, 0.5, 5.0])
        let idx = arr.argminArray()
        XCTAssertEqual(idx.dtype, .int64)
        XCTAssertEqual(idx.int64Data, [3])
    }

    func testArgsortReturnsInt64() {
        let arr = NDArray(shape: [5], data: [3.0, 1.0, 4.0, 1.0, 5.0])
        let indices = arr.argsort()
        XCTAssertEqual(indices.dtype, .int64)
        // Values at positions 1,3 are both 1.0; positions depend on stable sort
        XCTAssertEqual(indices.size, 5)
        // Verify sorted order is correct
        let sorted = indices.int64Data!.map { arr.real[Int($0)] }
        for i in 0..<sorted.count - 1 {
            XCTAssertLessThanOrEqual(sorted[i], sorted[i + 1])
        }
    }

    // MARK: - Bool NDArray Tests

    func testBoolCreation() {
        let arr = NDArray(shape: [4], boolData: [true, false, true, false])
        XCTAssertEqual(arr.dtype, .bool)
        XCTAssertEqual(arr.shape, [4])
        XCTAssertEqual(arr.boolData, [1, 0, 1, 0])
    }

    func testBoolRealAccessor() {
        let arr = NDArray(shape: [3], boolData: [true, false, true])
        XCTAssertEqual(arr.real, [1.0, 0.0, 1.0])
    }

    func testBoolLogicalAnd() {
        let a = NDArray(shape: [4], boolData: [true, true, false, false])
        let b = NDArray(shape: [4], boolData: [true, false, true, false])
        let c = a.logicalAnd(b)
        XCTAssertEqual(c.dtype, .bool)
        XCTAssertEqual(c.boolData, [1, 0, 0, 0])
    }

    func testBoolLogicalOr() {
        let a = NDArray(shape: [4], boolData: [true, true, false, false])
        let b = NDArray(shape: [4], boolData: [true, false, true, false])
        let c = a.logicalOr(b)
        XCTAssertEqual(c.dtype, .bool)
        XCTAssertEqual(c.boolData, [1, 1, 1, 0])
    }

    func testBoolLogicalNot() {
        let a = NDArray(shape: [3], boolData: [true, false, true])
        let c = a.logicalNot()
        XCTAssertEqual(c.dtype, .bool)
        XCTAssertEqual(c.boolData, [0, 1, 0])
    }

    func testBoolLogicalXor() {
        let a = NDArray(shape: [4], boolData: [true, true, false, false])
        let b = NDArray(shape: [4], boolData: [true, false, true, false])
        let c = a.logicalXor(b)
        XCTAssertEqual(c.dtype, .bool)
        XCTAssertEqual(c.boolData, [0, 1, 1, 0])
    }

    func testIsnaNReturnsBool() {
        let arr = NDArray(shape: [4], data: [1.0, Double.nan, 3.0, Double.nan])
        let result = arr.isnan()
        XCTAssertEqual(result.dtype, .bool)
        XCTAssertEqual(result.boolData, [0, 1, 0, 1])
    }

    func testIsinfReturnsBool() {
        let arr = NDArray(shape: [3], data: [1.0, Double.infinity, -Double.infinity])
        let result = arr.isinf()
        XCTAssertEqual(result.dtype, .bool)
        XCTAssertEqual(result.boolData, [0, 1, 1])
    }

    func testIsfiniteReturnsBool() {
        let arr = NDArray(shape: [3], data: [1.0, Double.infinity, Double.nan])
        let result = arr.isfinite()
        XCTAssertEqual(result.dtype, .bool)
        XCTAssertEqual(result.boolData, [1, 0, 0])
    }

    func testBoolPromotedToInt64() {
        let a = NDArray(shape: [3], boolData: [true, false, true])
        let b = NDArray(shape: [3], int64Data: [2, 3, 4])
        let c = a + b
        XCTAssertEqual(c.dtype, .int64)
        XCTAssertEqual(c.int64Data, [3, 3, 5])
    }

    func testBoolPromotedToFloat64() {
        let a = NDArray(shape: [3], boolData: [true, false, true])
        let b = NDArray(shape: [3], data: [1.5, 2.5, 3.5])
        let c = a + b
        XCTAssertEqual(c.dtype, .float64)
        XCTAssertEqual(c.real, [2.5, 2.5, 4.5])
    }

    // MARK: - Complex128 Regression Tests

    func testComplexRealImagStiillWork() {
        let arr = NDArray(shape: [2], dtype: .complex128, real: [1.0, 3.0], imag: [2.0, 4.0])
        XCTAssertEqual(arr.real, [1.0, 3.0])
        XCTAssertEqual(arr.imag, [2.0, 4.0])
        XCTAssertTrue(arr.isComplex)
    }

    func testComplexAbsStillWorks() {
        let arr = NDArray.complexArray(shape: [2], real: [3.0, 0.0], imag: [4.0, 5.0])
        let absArr = arr.abs()
        XCTAssertEqual(absArr.real[0], 5.0, accuracy: 1e-10)
        XCTAssertEqual(absArr.real[1], 5.0, accuracy: 1e-10)
    }

    func testComplexConj() {
        let arr = NDArray(shape: [2], dtype: .complex128, real: [1.0, 3.0], imag: [2.0, -4.0])
        let conj = arr.conjugate()
        XCTAssertEqual(conj.real, [1.0, 3.0])
        XCTAssertEqual(conj.imag, [-2.0, 4.0])
    }

    // MARK: - Date NDArray Tests

    func testDateCreation() {
        let ref = Date(timeIntervalSinceReferenceDate: 0)
        let d1 = Date(timeIntervalSinceReferenceDate: 86400)
        let arr = NDArray(shape: [2], dates: [ref, d1])
        XCTAssertEqual(arr.dtype, .date)
        XCTAssertEqual(arr.dateIntervals, [0.0, 86400.0])
    }

    func testDateRealAccessor() {
        let arr = NDArray(shape: [2], dateIntervals: [1000.0, 2000.0])
        XCTAssertEqual(arr.dtype, .date)
        XCTAssertEqual(arr.real, [1000.0, 2000.0])
    }

    func testDateAddInterval() {
        let arr = NDArray(shape: [3], dateIntervals: [0.0, 1000.0, 2000.0])
        // add 500 seconds to each date
        let shifted = try! arr.addInterval(500.0)
        XCTAssertEqual(shifted.dtype, .date)
        XCTAssertEqual(shifted.dateIntervals!, [500.0, 1500.0, 2500.0])
    }

    func testDateSubtractDates() throws {
        let a = NDArray(shape: [3], dateIntervals: [1000.0, 2000.0, 3000.0])
        let b = NDArray(shape: [3], dateIntervals: [0.0, 500.0, 1000.0])
        let diff = try a.subtractDates(b)
        XCTAssertEqual(diff.dtype, .float64)
        XCTAssertEqual(diff.real, [1000.0, 1500.0, 2000.0])
    }

    func testDateCompare() {
        let a = NDArray(shape: [3], dateIntervals: [1000.0, 2000.0, 3000.0])
        let b = NDArray(shape: [3], dateIntervals: [1000.0, 1999.0, 4000.0])
        let eq = a.equal(b)
        XCTAssertEqual(eq.real, [1.0, 0.0, 0.0])
        let lt = a.less(b)
        XCTAssertEqual(lt.real, [0.0, 0.0, 1.0])
    }

    func testDateInvalidMixThrows() {
        let dateArr = NDArray(shape: [2], dateIntervals: [1.0, 2.0])
        let complexArr = NDArray(shape: [2], dtype: .complex128, real: [1.0, 2.0], imag: [0.0, 0.0])
        XCTAssertThrowsError(try dateArr.addArray(complexArr))
    }

    func testDateInvalidIntMixThrows() {
        let dateArr = NDArray(shape: [2], dateIntervals: [1.0, 2.0])
        let intArr = NDArray(shape: [2], int64Data: [1, 2])
        XCTAssertThrowsError(try dateArr.addArray(intArr))
    }

    // MARK: - Float64/Complex128 backward compat

    func testFloat64BackwardCompat() {
        let arr = NDArray(shape: [3], data: [1.0, 2.0, 3.0])
        XCTAssertEqual(arr.dtype, .float64)
        XCTAssertEqual(arr.real, [1.0, 2.0, 3.0])
        XCTAssertNil(arr.imag)
        XCTAssertFalse(arr.isComplex)
    }

    func testComplex128BackwardCompat() {
        let arr = NDArray(shape: [2], dtype: .complex128, real: [1.0, 2.0], imag: [3.0, 4.0])
        XCTAssertEqual(arr.dtype, .complex128)
        XCTAssertEqual(arr.real, [1.0, 2.0])
        XCTAssertEqual(arr.imag, [2.0: 4.0].isEmpty ? [3.0, 4.0] : [3.0, 4.0])
        XCTAssertTrue(arr.isComplex)
    }

    func testDataAliasBackwardCompat() {
        let arr = NDArray(shape: [3], data: [1.0, 2.0, 3.0])
        XCTAssertEqual(arr.data, arr.real)
    }

    func testFloat64ArithmeticUnchanged() {
        let a = NDArray(shape: [3], data: [1.0, 2.0, 3.0])
        let b = NDArray(shape: [3], data: [4.0, 5.0, 6.0])
        let c = a + b
        XCTAssertEqual(c.dtype, .float64)
        XCTAssertEqual(c.real, [5.0, 7.0, 9.0])
    }

    func testIsNonZeroWithInt64() {
        let arr = NDArray(shape: [3], int64Data: [0, 1, -1])
        XCTAssertFalse(arr.isNonZero(at: 0))
        XCTAssertTrue(arr.isNonZero(at: 1))
        XCTAssertTrue(arr.isNonZero(at: 2))
    }

    func testIsNonZeroWithBool() {
        let arr = NDArray(shape: [2], boolData: [false, true])
        XCTAssertFalse(arr.isNonZero(at: 0))
        XCTAssertTrue(arr.isNonZero(at: 1))
    }
}
