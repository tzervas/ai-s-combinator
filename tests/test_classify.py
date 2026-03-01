"""Tests for S/K operation classifier."""

import pytest

from bwsk.classify import OpClass, classify_operation


class TestClassifyOperation:
    @pytest.mark.skip(reason="Not yet implemented")
    def test_classify_returns_opclass(self):
        result = classify_operation(None)
        assert isinstance(result, OpClass)

    @pytest.mark.skip(reason="Not yet implemented")
    def test_linear_is_s_type(self):
        """Linear projection (full-rank) should be S-type."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_relu_is_k_type(self):
        """ReLU should be K-type (erases negative values)."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_residual_is_s_type(self):
        """Residual connection should be S-type."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_dropout_is_k_type(self):
        """Dropout should be K-type."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_softmax_is_gray(self):
        """Softmax should be Gray (context-dependent)."""
        pass
