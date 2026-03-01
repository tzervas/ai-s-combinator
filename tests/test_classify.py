"""Tests for S/K operation classifier."""

import json

import pytest
import torch
import torch.nn as nn

from bwsk.classify import (
    ClassificationResult,
    ErasureBudgetReport,
    OpClass,
    classify_model,
    classify_operation,
)

# ---------------------------------------------------------------------------
# Unit tests: classify_operation for individual nn.Module types
# ---------------------------------------------------------------------------


class TestClassifyOperation:
    """Test classify_operation() with individual nn.Module instances."""

    def test_classify_returns_classification_result(self):
        result = classify_operation(nn.ReLU())
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.classification, OpClass)

    # --- Linear layers ---

    def test_linear_is_s_type(self):
        """Linear projection should be S-type."""
        result = classify_operation(nn.Linear(10, 10))
        assert result.classification == OpClass.S

    def test_lazy_linear_is_s_type(self):
        result = classify_operation(nn.LazyLinear(10))
        assert result.classification == OpClass.S

    def test_bilinear_is_s_type(self):
        result = classify_operation(nn.Bilinear(10, 10, 10))
        assert result.classification == OpClass.S

    # --- Convolutions ---

    def test_conv2d_stride1_is_s_type(self):
        result = classify_operation(nn.Conv2d(3, 16, 3, stride=1, padding=1))
        assert result.classification == OpClass.S

    def test_conv2d_stride2_is_k_type(self):
        result = classify_operation(nn.Conv2d(3, 16, 3, stride=2, padding=1))
        assert result.classification == OpClass.K

    def test_conv1d_stride1_is_s_type(self):
        result = classify_operation(nn.Conv1d(3, 16, 3, stride=1))
        assert result.classification == OpClass.S

    def test_conv_transpose2d_is_s_type(self):
        result = classify_operation(nn.ConvTranspose2d(16, 3, 3))
        assert result.classification == OpClass.S

    # --- Normalization ---

    def test_layer_norm_is_s_type(self):
        result = classify_operation(nn.LayerNorm(10))
        assert result.classification == OpClass.S

    def test_group_norm_is_s_type(self):
        result = classify_operation(nn.GroupNorm(2, 10))
        assert result.classification == OpClass.S

    def test_instance_norm_is_s_type(self):
        result = classify_operation(nn.InstanceNorm2d(10))
        assert result.classification == OpClass.S

    def test_batch_norm_train_is_k_type(self):
        """BatchNorm in training mode is K-type (cross-sample stats)."""
        bn = nn.BatchNorm2d(10)
        bn.train()
        result = classify_operation(bn)
        assert result.classification == OpClass.K

    def test_batch_norm_eval_is_s_type(self):
        """BatchNorm in eval mode is S-type (frozen stats)."""
        bn = nn.BatchNorm2d(10)
        bn.eval()
        result = classify_operation(bn)
        assert result.classification == OpClass.S

    # --- Activations ---

    def test_relu_is_k_type(self):
        """ReLU should be K-type (erases negative values)."""
        result = classify_operation(nn.ReLU())
        assert result.classification == OpClass.K

    def test_leaky_relu_is_s_type(self):
        result = classify_operation(nn.LeakyReLU())
        assert result.classification == OpClass.S

    def test_prelu_is_s_type(self):
        result = classify_operation(nn.PReLU())
        assert result.classification == OpClass.S

    def test_gelu_is_k_type(self):
        result = classify_operation(nn.GELU())
        assert result.classification == OpClass.K

    def test_silu_is_k_type(self):
        result = classify_operation(nn.SiLU())
        assert result.classification == OpClass.K

    def test_softplus_is_s_type(self):
        result = classify_operation(nn.Softplus())
        assert result.classification == OpClass.S

    def test_sigmoid_is_k_type(self):
        result = classify_operation(nn.Sigmoid())
        assert result.classification == OpClass.K

    def test_tanh_is_k_type(self):
        result = classify_operation(nn.Tanh())
        assert result.classification == OpClass.K

    def test_softmax_is_k_type(self):
        """Softmax should be K-type (reduces dimensionality by 1)."""
        result = classify_operation(nn.Softmax(dim=-1))
        assert result.classification == OpClass.K

    def test_elu_is_k_type(self):
        result = classify_operation(nn.ELU())
        assert result.classification == OpClass.K

    def test_mish_is_k_type(self):
        result = classify_operation(nn.Mish())
        assert result.classification == OpClass.K

    def test_hardswish_is_k_type(self):
        result = classify_operation(nn.Hardswish())
        assert result.classification == OpClass.K

    def test_hardsigmoid_is_k_type(self):
        result = classify_operation(nn.Hardsigmoid())
        assert result.classification == OpClass.K

    def test_log_softmax_is_k_type(self):
        result = classify_operation(nn.LogSoftmax(dim=-1))
        assert result.classification == OpClass.K

    # --- Pooling ---

    def test_max_pool2d_is_k_type(self):
        result = classify_operation(nn.MaxPool2d(2))
        assert result.classification == OpClass.K

    def test_avg_pool2d_is_k_type(self):
        result = classify_operation(nn.AvgPool2d(2))
        assert result.classification == OpClass.K

    def test_adaptive_avg_pool2d_is_k_type(self):
        result = classify_operation(nn.AdaptiveAvgPool2d(1))
        assert result.classification == OpClass.K

    def test_adaptive_max_pool2d_is_k_type(self):
        result = classify_operation(nn.AdaptiveMaxPool2d(1))
        assert result.classification == OpClass.K

    # --- Dropout ---

    def test_dropout_is_k_type(self):
        """Dropout should be K-type."""
        result = classify_operation(nn.Dropout(0.5))
        assert result.classification == OpClass.K

    def test_dropout2d_is_k_type(self):
        result = classify_operation(nn.Dropout2d(0.5))
        assert result.classification == OpClass.K

    def test_alpha_dropout_is_k_type(self):
        result = classify_operation(nn.AlphaDropout(0.5))
        assert result.classification == OpClass.K

    # --- Embedding ---

    def test_embedding_is_s_type(self):
        result = classify_operation(nn.Embedding(100, 64))
        assert result.classification == OpClass.S

    def test_embedding_bag_is_k_type(self):
        result = classify_operation(nn.EmbeddingBag(100, 64))
        assert result.classification == OpClass.K

    # --- Recurrent ---

    def test_lstm_is_gray(self):
        result = classify_operation(nn.LSTM(10, 20))
        assert result.classification == OpClass.GRAY

    def test_gru_is_gray(self):
        result = classify_operation(nn.GRU(10, 20))
        assert result.classification == OpClass.GRAY

    def test_rnn_is_gray(self):
        result = classify_operation(nn.RNN(10, 20))
        assert result.classification == OpClass.GRAY

    # --- Loss ---

    def test_cross_entropy_is_k_type(self):
        result = classify_operation(nn.CrossEntropyLoss())
        assert result.classification == OpClass.K

    def test_mse_loss_is_k_type(self):
        result = classify_operation(nn.MSELoss())
        assert result.classification == OpClass.K

    def test_l1_loss_is_k_type(self):
        result = classify_operation(nn.L1Loss())
        assert result.classification == OpClass.K

    # --- Attention ---

    def test_multihead_attention_is_gray(self):
        result = classify_operation(nn.MultiheadAttention(64, 8))
        assert result.classification == OpClass.GRAY

    # --- Confidence ---

    def test_relu_confidence_is_1(self):
        result = classify_operation(nn.ReLU())
        assert result.confidence == 1.0

    def test_linear_confidence_less_than_1(self):
        result = classify_operation(nn.Linear(10, 10))
        assert 0.8 <= result.confidence <= 1.0

    def test_result_has_rationale(self):
        result = classify_operation(nn.ReLU())
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0


class TestClassifyOperationCustomRules:
    """Test custom rule overrides."""

    def test_override_relu_to_s(self):
        result = classify_operation(nn.ReLU(), custom_rules={"nn.ReLU": OpClass.S})
        assert result.classification == OpClass.S
        assert result.confidence == 1.0

    def test_override_unknown_module(self):
        class MyCustomLayer(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        result = classify_operation(MyCustomLayer(), custom_rules={"MyCustomLayer": OpClass.S})
        assert result.classification == OpClass.S

    def test_unknown_module_defaults_to_gray(self):
        class MyCustomLayer(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        result = classify_operation(MyCustomLayer())
        assert result.classification == OpClass.GRAY


# ---------------------------------------------------------------------------
# Integration tests: classify_model with full models via torch.fx
# ---------------------------------------------------------------------------


class TestClassifyModel:
    """Test classify_model() with full models traced via torch.fx."""

    def test_simple_mlp(self):
        """MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.
        Expected: 3 S (Linear), 2 K (ReLU)."""

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(20, 20)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(20, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

        model = MLP()
        report = classify_model(model)

        assert isinstance(report, ErasureBudgetReport)
        assert report.s_count == 3  # 3 Linear layers
        assert report.k_count == 2  # 2 ReLU layers
        assert report.gray_count == 0
        assert report.total_ops == 5
        assert report.erasure_score == pytest.approx(2 / 5)

    def test_model_with_residual(self):
        """Model with residual connection: x + F(x)."""

        class ResBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.relu(self.fc(x))

        model = ResBlock()
        report = classify_model(model)

        assert isinstance(report, ErasureBudgetReport)
        # Linear is S, ReLU is K, add is S
        assert report.s_count >= 2  # Linear + add
        assert report.k_count >= 1  # ReLU

    def test_empty_sequential(self):
        """Empty Sequential model should produce an empty report."""
        model = nn.Sequential()
        report = classify_model(model)
        assert report.total_ops == 0
        assert report.erasure_score == 0.0

    def test_report_has_per_node_results(self):
        """Each node should have a ClassificationResult."""

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = SimpleModel()
        report = classify_model(model)

        assert len(report.per_node) > 0
        for result in report.per_node:
            assert isinstance(result, ClassificationResult)
            assert isinstance(result.classification, OpClass)

    def test_cnn_model(self):
        """CNN with stride-1 convs, ReLU, MaxPool, AdaptiveAvgPool, Linear."""

        class CNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
                self.relu2 = nn.ReLU()
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.gap(self.relu2(self.conv2(x)))
                return self.fc(self.flatten(x))

        model = CNN()
        report = classify_model(model)

        # S: Conv2d(s=1) x2, Flatten, Linear = 4
        # K: ReLU x2, MaxPool2d, AdaptiveAvgPool2d = 4
        assert report.s_count == 4
        assert report.k_count == 4
        assert report.gray_count == 0
        assert report.erasure_score == pytest.approx(0.5)

    def test_model_with_batchnorm_train(self):
        """BatchNorm in training mode should be classified as K."""

        class BNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.bn(self.fc(x))

        model = BNModel()
        model.train()
        report = classify_model(model)

        bn_results = [r for r in report.per_node if "bn" in r.op_name]
        assert len(bn_results) == 1
        assert bn_results[0].classification == OpClass.K

    def test_model_with_batchnorm_eval(self):
        """BatchNorm in eval mode should be classified as S."""

        class BNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.bn(self.fc(x))

        model = BNModel()
        model.eval()
        report = classify_model(model)

        bn_results = [r for r in report.per_node if "bn" in r.op_name]
        assert len(bn_results) == 1
        assert bn_results[0].classification == OpClass.S

    def test_model_with_functional_ops(self):
        """Functional ops (F.relu, torch.add) should be captured by torch.fx."""
        import torch.nn.functional as functional

        class FunctionalModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = functional.relu(self.fc1(x))
                return torch.add(h, self.fc2(x))

        model = FunctionalModel()
        report = classify_model(model)

        # S: Linear x2, torch.add = 3
        # K: F.relu = 1
        assert report.s_count == 3
        assert report.k_count == 1

    def test_custom_rules_in_model(self):
        """Custom rules should override default classifications in model."""

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.fc(x))

        model = MLP()
        report = classify_model(model, custom_rules={"nn.ReLU": OpClass.S})

        # ReLU should now be S instead of K
        relu_results = [
            r for r in report.per_node if "relu" in r.op_name.lower() or r.op_type == "nn.ReLU"
        ]
        assert len(relu_results) > 0
        assert all(r.classification == OpClass.S for r in relu_results)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test report serialization."""

    def test_to_dict_structure(self):
        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.fc(x))

        model = SimpleModel()
        report = classify_model(model)
        d = report.to_dict()

        assert "model_name" in d
        assert "total_ops" in d
        assert "s_count" in d
        assert "k_count" in d
        assert "gray_count" in d
        assert "erasure_score" in d
        assert "per_node" in d
        assert isinstance(d["per_node"], list)

    def test_to_json_is_valid_json(self):
        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = SimpleModel()
        report = classify_model(model)
        json_str = report.to_json()

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["total_ops"] == report.total_ops

    def test_per_layer_summary(self):
        """per_layer_summary groups results by layer prefix."""

        class TwoLayerMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
                self.layer2 = nn.Sequential(nn.Linear(20, 5), nn.ReLU())

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer2(self.layer1(x))

        model = TwoLayerMLP()
        report = classify_model(model)
        summary = report.per_layer_summary()

        assert isinstance(summary, dict)
        # Should have entries for layer1 and layer2 prefixes
        assert "layer1" in summary
        assert "layer2" in summary
        # Each layer has 1 S (Linear) and 1 K (ReLU)
        for layer_key in ("layer1", "layer2"):
            assert summary[layer_key]["s_count"] == 1
            assert summary[layer_key]["k_count"] == 1
            assert summary[layer_key]["gray_count"] == 0
            assert "erasure_score" in summary[layer_key]
