//! BWSK Core: Combinator primitives, S/K classification, and provenance tracking.
//!
//! This crate provides the pure computation layer for the BWSK framework,
//! with no ML framework dependencies. It implements:
//!
//! - **BWSK Combinators**: Type-safe combinator primitives (B, W, S, K) as
//!   generic Rust types.
//! - **S/K Classification**: Classify operations as information-preserving (S),
//!   information-erasing (K), or context-dependent (Gray).
//! - **Provenance**: Track information flow through computation graphs.

pub mod classify;
pub mod combinators;
pub mod provenance;
