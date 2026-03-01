//! S/K operation classification.
//!
//! Classifies operations as information-preserving (S-type),
//! information-erasing (K-type), or context-dependent (Gray).
//!
//! This is the Rust equivalent of Python's `bwsk.classify` module.

use std::collections::HashMap;

/// Classification of a neural network operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// Information-preserving: invertible, coordination-free.
    S,
    /// Information-erasing: synchronization point.
    K,
    /// Context-dependent: requires further analysis.
    Gray,
}

/// Result of classifying a single operation.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Name of the operation in the graph.
    pub op_name: String,
    /// Canonical type name (e.g., "nn.Linear").
    pub op_type: String,
    /// S/K/Gray classification.
    pub classification: OpClass,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Human-readable explanation.
    pub rationale: String,
}

/// Erasure budget report for a complete model.
#[derive(Debug, Clone)]
pub struct ErasureBudgetReport {
    /// Model name.
    pub model_name: String,
    /// Total number of operations.
    pub total_ops: usize,
    /// Number of S-type operations.
    pub s_count: usize,
    /// Number of K-type operations.
    pub k_count: usize,
    /// Number of Gray operations.
    pub gray_count: usize,
    /// Fraction of K-type operations (0.0 to 1.0).
    pub erasure_score: f64,
    /// Per-node classification results.
    pub per_node: Vec<ClassificationResult>,
}

impl ErasureBudgetReport {
    /// Fraction of S-type operations.
    pub fn s_ratio(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.s_count as f64 / self.total_ops as f64
        }
    }

    /// Fraction of K-type operations.
    pub fn k_ratio(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.k_count as f64 / self.total_ops as f64
        }
    }
}

/// Entry in the classification database.
#[derive(Debug, Clone)]
struct ClassEntry {
    class: OpClass,
    confidence: f64,
    rationale: String,
}

/// Classification database with default rules and user overrides.
pub struct Classifier {
    db: HashMap<String, ClassEntry>,
    custom_rules: HashMap<String, OpClass>,
}

impl Classifier {
    /// Create a new classifier with the default classification database.
    pub fn new() -> Self {
        let mut db = HashMap::new();

        // Linear layers
        Self::register(
            &mut db,
            "nn.Linear",
            OpClass::S,
            0.9,
            "Affine transform; invertible given full-rank weight.",
        );
        Self::register(
            &mut db,
            "nn.LazyLinear",
            OpClass::S,
            0.9,
            "Same as nn.Linear once materialized.",
        );
        Self::register(
            &mut db,
            "nn.Bilinear",
            OpClass::S,
            0.8,
            "Bilinear map; preserves info from both inputs.",
        );

        // Convolutions (stride=1 default)
        for name in &["nn.Conv1d", "nn.Conv2d", "nn.Conv3d"] {
            Self::register(
                &mut db,
                name,
                OpClass::S,
                0.9,
                "Convolution with stride=1; linear map with shared weights.",
            );
        }
        for name in &[
            "nn.ConvTranspose1d",
            "nn.ConvTranspose2d",
            "nn.ConvTranspose3d",
        ] {
            Self::register(
                &mut db,
                name,
                OpClass::S,
                0.9,
                "Transposed convolution; upsamples, no information loss.",
            );
        }

        // Normalization
        Self::register(
            &mut db,
            "nn.LayerNorm",
            OpClass::S,
            0.95,
            "Per-sample normalization; invertible given affine params.",
        );
        Self::register(
            &mut db,
            "nn.RMSNorm",
            OpClass::S,
            0.95,
            "Simpler LayerNorm variant; invertible given scale.",
        );
        Self::register(
            &mut db,
            "nn.GroupNorm",
            OpClass::S,
            0.9,
            "Per-group normalization; no cross-sample dependency.",
        );
        for name in &["nn.BatchNorm1d", "nn.BatchNorm2d", "nn.BatchNorm3d"] {
            Self::register(
                &mut db,
                name,
                OpClass::Gray,
                0.7,
                "BatchNorm: K in train mode, S in eval mode.",
            );
        }

        // Activations
        Self::register(
            &mut db,
            "nn.ReLU",
            OpClass::K,
            1.0,
            "max(0,x) erases all negative values.",
        );
        Self::register(
            &mut db,
            "nn.LeakyReLU",
            OpClass::S,
            0.95,
            "Bijective, no info loss.",
        );
        Self::register(
            &mut db,
            "nn.PReLU",
            OpClass::S,
            0.95,
            "Parametric LeakyReLU; bijective.",
        );
        Self::register(
            &mut db,
            "nn.GELU",
            OpClass::K,
            0.9,
            "Non-monotonic near zero; not injective.",
        );
        Self::register(
            &mut db,
            "nn.SiLU",
            OpClass::K,
            0.9,
            "Non-monotonic for x<0; not injective.",
        );
        Self::register(
            &mut db,
            "nn.Softplus",
            OpClass::S,
            0.95,
            "Strictly monotonic and invertible.",
        );
        Self::register(
            &mut db,
            "nn.Sigmoid",
            OpClass::K,
            0.95,
            "Saturates, practically K-type.",
        );
        Self::register(
            &mut db,
            "nn.Tanh",
            OpClass::K,
            0.95,
            "Saturates, practically K-type.",
        );
        Self::register(
            &mut db,
            "nn.Softmax",
            OpClass::K,
            1.0,
            "Reduces dimensionality by 1.",
        );

        // Pooling
        for name in &["nn.MaxPool1d", "nn.MaxPool2d", "nn.MaxPool3d"] {
            Self::register(
                &mut db,
                name,
                OpClass::K,
                1.0,
                "Selects maximum from window; all other values erased.",
            );
        }
        for name in &["nn.AvgPool1d", "nn.AvgPool2d", "nn.AvgPool3d"] {
            Self::register(
                &mut db,
                name,
                OpClass::K,
                1.0,
                "Averages window; individual values irrecoverable.",
            );
        }

        // Dropout
        for name in &["nn.Dropout", "nn.Dropout1d", "nn.Dropout2d", "nn.Dropout3d"] {
            Self::register(
                &mut db,
                name,
                OpClass::K,
                1.0,
                "Randomly zeroes elements; erases information stochastically.",
            );
        }

        // Embedding
        Self::register(
            &mut db,
            "nn.Embedding",
            OpClass::S,
            1.0,
            "Lookup table; injective mapping.",
        );
        Self::register(
            &mut db,
            "nn.EmbeddingBag",
            OpClass::K,
            0.9,
            "Aggregates embeddings; individual values lost.",
        );

        // Misc
        Self::register(
            &mut db,
            "nn.Identity",
            OpClass::S,
            1.0,
            "Identity function; no transformation.",
        );
        Self::register(
            &mut db,
            "nn.Flatten",
            OpClass::S,
            1.0,
            "Reshape; bijection on element ordering.",
        );

        // Loss functions
        for name in &[
            "nn.CrossEntropyLoss",
            "nn.MSELoss",
            "nn.L1Loss",
            "nn.BCELoss",
            "nn.NLLLoss",
            "nn.KLDivLoss",
        ] {
            Self::register(
                &mut db,
                name,
                OpClass::K,
                1.0,
                "Loss function; reduces to scalar, maximal erasure.",
            );
        }

        Self {
            db,
            custom_rules: HashMap::new(),
        }
    }

    fn register(
        db: &mut HashMap<String, ClassEntry>,
        name: &str,
        class: OpClass,
        confidence: f64,
        rationale: &str,
    ) {
        db.insert(
            name.to_string(),
            ClassEntry {
                class,
                confidence,
                rationale: rationale.to_string(),
            },
        );
    }

    /// Add a custom classification rule (overrides defaults).
    pub fn add_rule(&mut self, op_name: &str, class: OpClass) {
        self.custom_rules.insert(op_name.to_string(), class);
    }

    /// Classify a single operation by its canonical name.
    pub fn classify(&self, op_type: &str) -> ClassificationResult {
        // Step 1: User override
        if let Some(&class) = self.custom_rules.get(op_type) {
            return ClassificationResult {
                op_name: String::new(),
                op_type: op_type.to_string(),
                classification: class,
                confidence: 1.0,
                rationale: "User override".to_string(),
            };
        }

        // Step 2: Database lookup
        if let Some(entry) = self.db.get(op_type) {
            return ClassificationResult {
                op_name: String::new(),
                op_type: op_type.to_string(),
                classification: entry.class,
                confidence: entry.confidence,
                rationale: entry.rationale.clone(),
            };
        }

        // Step 3: Default to Gray
        ClassificationResult {
            op_name: String::new(),
            op_type: op_type.to_string(),
            classification: OpClass::Gray,
            confidence: 0.0,
            rationale: format!("Unrecognized operation: {}; defaulting to Gray.", op_type),
        }
    }

    /// Classify a list of operations and produce an erasure budget report.
    pub fn classify_ops(&self, model_name: &str, ops: &[&str]) -> ErasureBudgetReport {
        let mut results = Vec::new();
        for (i, op_type) in ops.iter().enumerate() {
            let mut result = self.classify(op_type);
            result.op_name = format!("op_{}", i);
            results.push(result);
        }

        let s_count = results
            .iter()
            .filter(|r| r.classification == OpClass::S)
            .count();
        let k_count = results
            .iter()
            .filter(|r| r.classification == OpClass::K)
            .count();
        let gray_count = results
            .iter()
            .filter(|r| r.classification == OpClass::Gray)
            .count();
        let total_ops = results.len();
        let erasure_score = if total_ops > 0 {
            k_count as f64 / total_ops as f64
        } else {
            0.0
        };

        ErasureBudgetReport {
            model_name: model_name.to_string(),
            total_ops,
            s_count,
            k_count,
            gray_count,
            erasure_score,
            per_node: results,
        }
    }
}

impl Default for Classifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_is_k() {
        let c = Classifier::new();
        let result = c.classify("nn.ReLU");
        assert_eq!(result.classification, OpClass::K);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_linear_is_s() {
        let c = Classifier::new();
        let result = c.classify("nn.Linear");
        assert_eq!(result.classification, OpClass::S);
    }

    #[test]
    fn test_unknown_is_gray() {
        let c = Classifier::new();
        let result = c.classify("nn.CustomUnknown");
        assert_eq!(result.classification, OpClass::Gray);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_custom_rule_overrides() {
        let mut c = Classifier::new();
        c.add_rule("nn.ReLU", OpClass::S);
        let result = c.classify("nn.ReLU");
        assert_eq!(result.classification, OpClass::S);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_erasure_budget_report() {
        let c = Classifier::new();
        let ops = vec!["nn.Linear", "nn.ReLU", "nn.Linear", "nn.ReLU", "nn.Linear"];
        let report = c.classify_ops("MLP", &ops);

        assert_eq!(report.total_ops, 5);
        assert_eq!(report.s_count, 3);
        assert_eq!(report.k_count, 2);
        assert_eq!(report.gray_count, 0);
        assert!((report.erasure_score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_batch_norm_is_gray() {
        let c = Classifier::new();
        let result = c.classify("nn.BatchNorm2d");
        assert_eq!(result.classification, OpClass::Gray);
    }

    #[test]
    fn test_s_ratio() {
        let c = Classifier::new();
        let report = c.classify_ops("test", &["nn.Linear", "nn.ReLU"]);
        assert!((report.s_ratio() - 0.5).abs() < 1e-6);
    }
}
