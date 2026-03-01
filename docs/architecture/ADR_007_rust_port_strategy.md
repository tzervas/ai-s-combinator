# ADR-007: Rust Port Strategy

## Status
Accepted

## Context
Phase 4 ports the BWSK framework to Rust for performance, type safety, and single-binary deployment. Need to decide crate structure, ML framework integration, and what to port.

## Decision

### Two-crate architecture:
1. **bwsk-core**: Pure Rust, zero ML dependencies. Contains combinators, classifier, and provenance tracker. Can be used standalone for architecture analysis.
2. **bwsk-burn**: Integrates with [Burn](https://burn.dev/) ML framework. Provides BWSK modules (BLinear, SResidual, KRelu, BwskMlp) that compile to Burn modules with GPU support via CubeCL.

### Design choices:

1. **Combinators as functions, not traits**: Rust combinators (`b`, `w`, `s`, `k`) are generic functions returning closures. This is more idiomatic than trait-based composition and leverages Rust's zero-cost closures.

2. **Burn for ML integration**: Chose Burn over tch-rs (PyTorch bindings) because Burn is pure Rust with native GPU backends (CubeCL for CUDA/Vulkan/Metal), enabling single-binary deployment without libtorch.

3. **Classifier uses string-based op names**: Same canonical naming as Python (`"nn.Linear"`, `"nn.ReLU"`) for cross-language consistency. The Rust classifier can validate Python-classified models.

4. **Concrete module types (BLinear, SResidual)** rather than generic `BModule<F, G>`: Burn's derive macro system works best with concrete types. Generic combinator composition uses the function-level API in bwsk-core.

## Alternatives Considered

1. **tch-rs (PyTorch C++ bindings)**: Rejected because it requires libtorch shared library, defeating the single-binary deployment goal. Also, tch-rs is a thin wrapper without Rust-native optimizations.

2. **candle (Hugging Face Rust ML)**: Considered but Burn was chosen for its more mature module system and CubeCL GPU backend which provides cross-platform GPU support without vendor lock-in.

3. **Trait-based combinator system**: `trait Combinator { type Input; type Output; fn apply(&self, x: Input) -> Output; }` — rejected because the trait system becomes complex with higher-kinded types needed for combinator composition. Function-based API is simpler and more ergonomic.
