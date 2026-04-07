# Platform Support

Supported platforms and Accelerate framework usage.

## Supported Platforms

ArraySwift requires Apple platforms with the Accelerate framework:

| Platform | Minimum Version |
|---|---|
| macOS | 12.0+ |
| iOS | 15.0+ |
| tvOS | 15.0+ |
| watchOS | 8.0+ |
| visionOS | 1.0+ |

**Swift version**: 5.9+

## Accelerate Framework Usage

ArraySwift uses Apple's Accelerate framework (vDSP and vForce) for vectorized operations where available:

**Accelerated operations** (use vDSP/vForce):
- Element-wise arithmetic (add, subtract, multiply, divide)
- Complex arithmetic (vDSP_zvadd, vDSP_zvmul, etc.)
- Global reductions (sum, mean via vDSP_sveD, vDSP_meanvD)
- Math functions (sin, cos, exp, log, sqrt via vForce)
- Negation, absolute value

**Scalar loop operations**:
- Axis reductions (iterate over strided elements)
- Power, modulo
- Complex math (log, sqrt, trig for complex inputs)
- Comparisons and logical operations
- Broadcasting (materializes full arrays)

## Linux and Windows

ArraySwift is not currently supported on Linux or Windows due to its dependency on the Accelerate framework. There are no immediate plans for cross-platform support.
