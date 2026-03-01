//! BWSK combinator primitives.
//!
//! Pure combinator implementations using Rust's type system.
//! These operate on generic types — no tensor/ML dependencies.
//!
//! - `B f g x = f(g(x))` — Composition
//! - `W f x = f(x, x)` — Self-application / weight sharing
//! - `S f g x = f(x) + g(x)` — Fan-out and combine
//! - `K f x y = f(x)` — Erasure (y discarded)

/// Composition combinator: B f g x = f(g(x)).
///
/// Chains two functions sequentially. Maps to layer stacking in neural networks.
/// S-type: no information is lost in composition.
pub fn b<A, B, C>(f: impl Fn(B) -> C, g: impl Fn(A) -> B) -> impl Fn(A) -> C {
    move |x| f(g(x))
}

/// Self-application combinator: W f x = f(x, x).
///
/// Feeds the same input to both arguments of a binary function.
/// Maps to weight sharing and self-attention.
/// S-type: input is duplicated, not erased.
pub fn w<A: Clone, C>(f: impl Fn(A, A) -> C) -> impl Fn(A) -> C {
    move |x| {
        let x2 = x.clone();
        f(x, x2)
    }
}

/// Fan-out combinator: S f g x = combine(f(x), g(x)).
///
/// Applies both f and g to the same input, then combines results.
/// Maps to residual connections and multi-head patterns.
/// S-type: all information from x is preserved across both branches.
pub fn s<A: Clone, B, C, D>(
    f: impl Fn(A) -> B,
    g: impl Fn(A) -> C,
    combine: impl Fn(B, C) -> D,
) -> impl Fn(A) -> D {
    move |x| {
        let x2 = x.clone();
        combine(f(x), g(x2))
    }
}

/// Erasure combinator: K f x y = f(x).
///
/// Applies f to x, explicitly discarding y.
/// Maps to masking, dropout, pooling.
/// K-type: y is erased.
pub fn k<A, B, C>(f: impl Fn(A) -> B) -> impl Fn(A, C) -> B {
    move |x, _y| f(x)
}

/// Combinator kind, used for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombinatorKind {
    /// B: Composition (sequential)
    B,
    /// W: Self-application (weight sharing)
    W,
    /// S: Fan-out and combine (residual)
    S,
    /// K: Erasure (information loss)
    K,
}

impl CombinatorKind {
    /// Whether this combinator preserves information (S-type in classification).
    pub fn is_preserving(&self) -> bool {
        matches!(
            self,
            CombinatorKind::B | CombinatorKind::W | CombinatorKind::S
        )
    }

    /// Whether this combinator erases information (K-type in classification).
    pub fn is_erasing(&self) -> bool {
        matches!(self, CombinatorKind::K)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b_composition() {
        let inc = |x: i32| x + 1;
        let double = |x: i32| x * 2;
        let composed = b(inc, double);
        assert_eq!(composed(3), 7); // inc(double(3)) = inc(6) = 7
    }

    #[test]
    fn test_w_self_application() {
        let add = |x: i32, y: i32| x + y;
        let shared = w(add);
        assert_eq!(shared(3), 6); // add(3, 3) = 6
    }

    #[test]
    fn test_s_fan_out() {
        let inc = |x: i32| x + 1;
        let double = |x: i32| x * 2;
        let add = |a: i32, b: i32| a + b;
        let fan_out = s(inc, double, add);
        assert_eq!(fan_out(3), 10); // inc(3) + double(3) = 4 + 6 = 10
    }

    #[test]
    fn test_k_erasure() {
        let identity = |x: i32| x;
        let erased = k(identity);
        assert_eq!(erased(42, "discarded"), 42);
    }

    #[test]
    fn test_combinator_kind_preserving() {
        assert!(CombinatorKind::B.is_preserving());
        assert!(CombinatorKind::W.is_preserving());
        assert!(CombinatorKind::S.is_preserving());
        assert!(!CombinatorKind::K.is_preserving());
    }

    #[test]
    fn test_combinator_kind_erasing() {
        assert!(!CombinatorKind::B.is_erasing());
        assert!(CombinatorKind::K.is_erasing());
    }

    #[test]
    fn test_composition_chain() {
        // B(inc, B(double, inc))(3) = inc(double(inc(3))) = inc(double(4)) = inc(8) = 9
        let inc = |x: i32| x + 1;
        let double = |x: i32| x * 2;
        let inner = b(double, inc);
        let outer = b(|x: i32| x + 1, inner);
        assert_eq!(outer(3), 9);
    }
}
