# API Stability

Versioning policy and compatibility guarantees.

## Semantic Versioning

ArraySwift follows [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0 → 2.0.0): Breaking API changes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Patch** (0.1.0 → 0.1.1): Bug fixes, backward compatible

## Pre-1.0 Status

ArraySwift is currently in pre-release (0.x). During this phase:

- Minor version bumps may include breaking changes
- The API surface is not yet frozen
- Feedback on API design is welcome via GitHub issues

## Stability Tiers

Once 1.0 is released, public APIs will be categorized:

- **Stable**: Will not change without a major version bump and deprecation period.
- **Provisional**: May change in minor versions. Documented as such in DocC.

## Deprecation Policy

When an API must change:

1. The old API is marked `@available(*, deprecated, renamed:)` or `@available(*, deprecated, message:)`
2. The replacement is available in the same release
3. The deprecated API is removed in the next major version

## Source Compatibility

ArraySwift targets Swift 5.9+ and maintains source compatibility within the same major version across Swift releases.
