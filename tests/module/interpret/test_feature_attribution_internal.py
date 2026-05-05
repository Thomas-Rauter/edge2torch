import pytest
import torch
from torch import nn

from edge2torch.interpret.feature_attribution import (
    _build_feature_interpreter,
    _normalize_feature_attributions,
    _validate_feature_attributions,
    run_feature_attribution,
)
from edge2torch.utils.errors import Edge2TorchError


class _Artifact:
    def __init__(self, backend="feedforward"):
        self.backend = backend


class _TinyModel(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


def test_run_feature_attribution_rejects_unsupported_backend():
    model = _TinyModel()
    artifact = _Artifact(backend="unknown")

    with pytest.raises(
        Edge2TorchError,
        match="Unsupported backend",
    ):
        run_feature_attribution(
            model=model,
            artifact=artifact,
            inputs=torch.randn(2, 2),
            sample_names=["sample_1", "sample_2"],
            feature_names=["gene_a", "gene_b"],
            method="integrated_gradients",
            constructor_kwargs={},
            attribute_kwargs={},
        )


def test_build_feature_interpreter_rejects_unknown_method():
    model = _TinyModel()

    with pytest.raises(
        Edge2TorchError,
        match="not supported for target='features'",
    ):
        _build_feature_interpreter(
            method="not_a_method",
            model=model,
            constructor_kwargs={},
        )


def test_normalize_feature_attributions_squeezes_singleton_output_dimension():
    attributions = torch.randn(2, 1, 3)

    result = _normalize_feature_attributions(attributions)

    assert result.shape == (2, 3)
    assert torch.equal(result, attributions.squeeze(1))


def test_normalize_feature_attributions_keeps_2d_tensor_unchanged():
    attributions = torch.randn(2, 3)

    result = _normalize_feature_attributions(attributions)

    assert result is attributions


def test_normalize_feature_attributions_keeps_non_singleton_3d_tensor():
    attributions = torch.randn(2, 2, 3)

    result = _normalize_feature_attributions(attributions)

    assert result is attributions


def test_validate_feature_attributions_rejects_non_2d_tensor():
    attributions = torch.randn(2, 1, 3)

    with pytest.raises(
        Edge2TorchError,
        match="shape \\(n_examples, n_features\\)",
    ):
        _validate_feature_attributions(
            attributions=attributions,
            sample_names=["sample_1", "sample_2"],
            feature_names=["gene_a", "gene_b", "gene_c"],
        )


def test_validate_feature_attributions_rejects_wrong_row_count():
    attributions = torch.randn(3, 2)

    with pytest.raises(
        Edge2TorchError,
        match="row count does not match sample count",
    ):
        _validate_feature_attributions(
            attributions=attributions,
            sample_names=["sample_1", "sample_2"],
            feature_names=["gene_a", "gene_b"],
        )


def test_validate_feature_attributions_rejects_wrong_column_count():
    attributions = torch.randn(2, 3)

    with pytest.raises(
        Edge2TorchError,
        match="column count does not match feature count",
    ):
        _validate_feature_attributions(
            attributions=attributions,
            sample_names=["sample_1", "sample_2"],
            feature_names=["gene_a", "gene_b"],
        )


def test_validate_feature_attributions_accepts_matching_shape():
    attributions = torch.randn(2, 2)

    _validate_feature_attributions(
        attributions=attributions,
        sample_names=["sample_1", "sample_2"],
        feature_names=["gene_a", "gene_b"],
    )
