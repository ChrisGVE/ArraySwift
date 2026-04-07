# Thread Safety

Concurrency guarantees and safe usage patterns for NDArray.

## Sendable Conformance

`NDArray` conforms to `Sendable` because it is a value type (struct) backed by Swift `Array` storage, which uses copy-on-write semantics. This means:

- **Safe to send across concurrency boundaries**: You can pass an `NDArray` to a `Task`, actor, or `@Sendable` closure.
- **Mutations are isolated**: When you mutate an `NDArray`, Swift's copy-on-write ensures the original is not affected.

## Safe Patterns

```swift
// Safe: each task gets its own copy
let shared = NDArray.ones([100, 100])
Task {
    let result = shared + NDArray.ones([100, 100])
    // 'shared' is unchanged
}
```

## Unsafe Patterns to Avoid

Since `NDArray` is a struct, there is no shared mutable state by default. However, be careful with:

- **Storing NDArrays in reference types**: If you store an `NDArray` in a class or actor, ensure proper synchronization when mutating it from multiple contexts.
- **Large array copies**: Sending large arrays across concurrency boundaries triggers copy-on-write only when mutated, but the initial send is O(1).

## Performance Considerations

- Copy-on-write means reading a shared array is free (no copy until mutation).
- Mutating a shared array triggers a full copy of the backing storage.
- For compute-heavy parallel workloads, partition the data first and assign each partition to a separate task.
