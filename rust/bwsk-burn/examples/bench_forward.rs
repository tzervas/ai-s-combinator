//! Forward pass benchmark for Burn BWSK modules.
//!
//! Runs BwskMlp forward passes at varying sizes and reports throughput.
//! Also verifies that the S/K analysis matches expected classifications.
//!
//! Usage:
//!     cargo run --example bench_forward --release

use std::time::Instant;

use burn::backend::NdArray;
use burn::prelude::*;
use bwsk_burn::{BLinear, BwskMlp, KRelu, SResidual};
use bwsk_core::classify::OpClass;

type B = NdArray;

fn bench_bwsk_mlp(batch: usize, in_dim: usize, hidden: usize, out_dim: usize, iters: usize) {
    let device = Default::default();
    let model = BwskMlp::<B>::new(&device, in_dim, hidden, out_dim);
    let x = Tensor::<B, 2>::zeros([batch, in_dim], &device);

    // Warmup
    for _ in 0..5 {
        let _ = model.forward(x.clone());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(x.clone());
    }
    let elapsed = start.elapsed();

    let us_per_iter = elapsed.as_micros() as f64 / iters as f64;
    let samples_per_sec = (batch as f64 * iters as f64) / elapsed.as_secs_f64();

    println!(
        "  BwskMlp [{batch}x{in_dim}] -> [{batch}x{out_dim}] (hidden={hidden}): \
         {us_per_iter:.1}µs/iter, {samples_per_sec:.0} samples/s"
    );
}

fn bench_b_linear(batch: usize, in_dim: usize, hidden: usize, out_dim: usize, iters: usize) {
    let device = Default::default();
    let model = BLinear::<B>::new(&device, in_dim, hidden, out_dim);
    let x = Tensor::<B, 2>::zeros([batch, in_dim], &device);

    // Warmup
    for _ in 0..5 {
        let _ = model.forward(x.clone());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(x.clone());
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / iters as f64;

    println!(
        "  BLinear  [{batch}x{in_dim}] -> [{batch}x{out_dim}] (hidden={hidden}): \
         {us_per_iter:.1}µs/iter"
    );
}

fn bench_s_residual(batch: usize, dim: usize, iters: usize) {
    let device = Default::default();
    let model = SResidual::<B>::new(&device, dim);
    let x = Tensor::<B, 2>::zeros([batch, dim], &device);

    // Warmup
    for _ in 0..5 {
        let _ = model.forward(x.clone());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(x.clone());
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / iters as f64;

    println!("  SResidual [{batch}x{dim}] -> [{batch}x{dim}]: {us_per_iter:.1}µs/iter");
}

fn bench_k_relu(batch: usize, dim: usize, iters: usize) {
    let device: <B as Backend>::Device = Default::default();
    let model = KRelu::new();
    let x = Tensor::<B, 2>::zeros([batch, dim], &device);

    // Warmup
    for _ in 0..5 {
        let _ = model.forward(x.clone());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(x.clone());
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / iters as f64;

    println!("  KRelu     [{batch}x{dim}] -> [{batch}x{dim}]: {us_per_iter:.1}µs/iter");
}

fn verify_classifications() {
    println!("\n--- Classification Verification ---");

    let device = Default::default();

    let b_linear = BLinear::<B>::new(&device, 10, 20, 5);
    assert_eq!(b_linear.classification(), OpClass::S);
    println!("  BLinear:    S (correct)");

    let s_residual = SResidual::<B>::new(&device, 10);
    assert_eq!(s_residual.classification(), OpClass::S);
    println!("  SResidual:  S (correct)");

    let k_relu = KRelu::new();
    assert_eq!(k_relu.classification(), OpClass::K);
    println!("  KRelu:      K (correct)");

    let mlp = BwskMlp::<B>::new(&device, 10, 20, 5);
    let analysis = mlp.analyze();
    let s_count = analysis.iter().filter(|(_, c)| *c == OpClass::S).count();
    let k_count = analysis.iter().filter(|(_, c)| *c == OpClass::K).count();
    assert_eq!(s_count, 3);
    assert_eq!(k_count, 2);
    println!(
        "  BwskMlp:    {} layers (S={}, K={}) — matches Python",
        analysis.len(),
        s_count,
        k_count,
    );
}

fn main() {
    println!("=== BWSK Burn Forward Pass Benchmark ===\n");
    println!("Backend: NdArray (CPU)");

    let iters = 100;

    // Small models (typical transformer hidden sizes)
    println!("\n--- Small (hidden=128) ---");
    bench_bwsk_mlp(32, 64, 128, 32, iters);
    bench_b_linear(32, 64, 128, 32, iters);
    bench_s_residual(32, 128, iters);
    bench_k_relu(32, 128, iters);

    // Medium models
    println!("\n--- Medium (hidden=512) ---");
    bench_bwsk_mlp(16, 256, 512, 128, iters);
    bench_b_linear(16, 256, 512, 128, iters);
    bench_s_residual(16, 512, iters);
    bench_k_relu(16, 512, iters);

    // Large models (GPT-2 scale)
    println!("\n--- Large (hidden=1024) ---");
    bench_bwsk_mlp(8, 768, 1024, 768, iters);
    bench_b_linear(8, 768, 1024, 768, iters);
    bench_s_residual(8, 1024, iters);
    bench_k_relu(8, 1024, iters);

    // Extra large (1B-scale dims)
    println!("\n--- Extra Large (hidden=2048) ---");
    bench_bwsk_mlp(4, 2048, 2048, 2048, iters / 10);
    bench_b_linear(4, 2048, 2048, 2048, iters / 10);
    bench_s_residual(4, 2048, iters / 10);
    bench_k_relu(4, 2048, iters / 10);

    verify_classifications();

    println!("\nBenchmark complete.");
}
