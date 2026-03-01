//! BWSK Burn: Integration with the Burn ML framework.
//!
//! Provides BWSK combinator modules that compile to Burn modules with
//! tensor computation and GPU support. Each combinator is a Burn `Module`
//! that can be composed into larger architectures.
//!
//! # Architecture
//!
//! - `BModule<B>`: Sequential composition `f(g(x))`
//! - `SModule<B>`: Fan-out and combine `f(x) + g(x)` (residual pattern)
//! - `KModule<B>`: Erasure wrapper marking K-type operations
//!
//! These modules self-report their S/K classification via `classification()`,
//! enabling compile-time architecture analysis.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use bwsk_core::classify::OpClass;

/// Composition combinator as a Burn module: B f g x = f(g(x)).
///
/// Chains two linear layers sequentially. S-type because no information
/// is lost in linear composition (given full-rank weights).
#[derive(Module, Debug)]
pub struct BLinear<B: Backend> {
    f: Linear<B>,
    g: Linear<B>,
}

impl<B: Backend> BLinear<B> {
    /// Create a new B-composition of two linear layers.
    pub fn new(device: &B::Device, in_dim: usize, hidden: usize, out_dim: usize) -> Self {
        Self {
            g: LinearConfig::new(in_dim, hidden).init(device),
            f: LinearConfig::new(hidden, out_dim).init(device),
        }
    }

    /// Forward pass: f(g(x)).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.f.forward(self.g.forward(x))
    }

    /// S/K classification of this combinator.
    pub fn classification(&self) -> OpClass {
        OpClass::S
    }
}

/// Fan-out combinator as a Burn module: S f g x = f(x) + g(x).
///
/// Applies two branches to the same input and combines by addition.
/// Models residual connections. S-type because addition is invertible
/// given one operand.
#[derive(Module, Debug)]
pub struct SResidual<B: Backend> {
    transform: Linear<B>,
    activation: Relu,
}

impl<B: Backend> SResidual<B> {
    /// Create a new residual block: x + relu(linear(x)).
    pub fn new(device: &B::Device, dim: usize) -> Self {
        Self {
            transform: LinearConfig::new(dim, dim).init(device),
            activation: Relu::new(),
        }
    }

    /// Forward pass: x + relu(linear(x)).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = self.activation.forward(self.transform.forward(x.clone()));
        x + residual
    }

    /// S/K classification of this combinator.
    pub fn classification(&self) -> OpClass {
        OpClass::S
    }
}

/// Erasure combinator as a Burn module wrapper.
///
/// Wraps a module that performs information-erasing computation
/// (e.g., ReLU, pooling). K-type because information is lost.
#[derive(Module, Debug, Clone)]
pub struct KRelu {
    relu: Relu,
}

impl KRelu {
    /// Create a new K-type ReLU wrapper.
    pub fn new() -> Self {
        Self { relu: Relu::new() }
    }

    /// Forward pass: relu(x).
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.relu.forward(x)
    }

    /// S/K classification of this combinator.
    pub fn classification(&self) -> OpClass {
        OpClass::K
    }
}

impl Default for KRelu {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple MLP built using BWSK combinators.
///
/// Architecture: Linear -> K(ReLU) -> Linear -> K(ReLU) -> Linear
/// BWSK expression: B(Linear, B(K(ReLU), B(Linear, B(K(ReLU), Linear))))
#[derive(Module, Debug)]
pub struct BwskMlp<B: Backend> {
    layer1: Linear<B>,
    relu1: KRelu,
    layer2: Linear<B>,
    relu2: KRelu,
    layer3: Linear<B>,
}

impl<B: Backend> BwskMlp<B> {
    /// Create a new BWSK MLP.
    pub fn new(device: &B::Device, in_dim: usize, hidden: usize, out_dim: usize) -> Self {
        Self {
            layer1: LinearConfig::new(in_dim, hidden).init(device),
            relu1: KRelu::new(),
            layer2: LinearConfig::new(hidden, hidden).init(device),
            relu2: KRelu::new(),
            layer3: LinearConfig::new(hidden, out_dim).init(device),
        }
    }

    /// Forward pass through the MLP.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(x);
        let x = self.relu1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.relu2.forward(x);
        self.layer3.forward(x)
    }

    /// Analyze the S/K composition of this model.
    pub fn analyze(&self) -> Vec<(&str, OpClass)> {
        vec![
            ("layer1", OpClass::S),
            ("relu1", OpClass::K),
            ("layer2", OpClass::S),
            ("relu2", OpClass::K),
            ("layer3", OpClass::S),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_b_linear_forward() {
        let device = Default::default();
        let model = BLinear::<TestBackend>::new(&device, 10, 20, 5);
        let x = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        let out = model.forward(x);
        assert_eq!(out.dims(), [4, 5]);
    }

    #[test]
    fn test_b_linear_classification() {
        let device = Default::default();
        let model = BLinear::<TestBackend>::new(&device, 10, 20, 5);
        assert_eq!(model.classification(), OpClass::S);
    }

    #[test]
    fn test_s_residual_forward() {
        let device = Default::default();
        let model = SResidual::<TestBackend>::new(&device, 10);
        let x = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        let out = model.forward(x);
        assert_eq!(out.dims(), [4, 10]);
    }

    #[test]
    fn test_s_residual_classification() {
        let device = Default::default();
        let model = SResidual::<TestBackend>::new(&device, 10);
        assert_eq!(model.classification(), OpClass::S);
    }

    #[test]
    fn test_k_relu_classification() {
        let k = KRelu::new();
        assert_eq!(k.classification(), OpClass::K);
    }

    #[test]
    fn test_k_relu_forward() {
        let device = Default::default();
        let k = KRelu::new();
        let x = Tensor::<TestBackend, 2>::from_data([[1.0, -1.0, 2.0, -2.0]], &device);
        let out = k.forward(x);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_bwsk_mlp_forward() {
        let device = Default::default();
        let model = BwskMlp::<TestBackend>::new(&device, 10, 20, 5);
        let x = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        let out = model.forward(x);
        assert_eq!(out.dims(), [4, 5]);
    }

    #[test]
    fn test_bwsk_mlp_analyze() {
        let device = Default::default();
        let model = BwskMlp::<TestBackend>::new(&device, 10, 20, 5);
        let analysis = model.analyze();
        assert_eq!(analysis.len(), 5);

        let s_count = analysis.iter().filter(|(_, c)| *c == OpClass::S).count();
        let k_count = analysis.iter().filter(|(_, c)| *c == OpClass::K).count();
        assert_eq!(s_count, 3);
        assert_eq!(k_count, 2);
    }
}
