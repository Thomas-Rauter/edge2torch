import pytest
import torch
from torch import nn

from edge2torch.interpret.feedforward_node_attribution import (
    _build_feedforward_layer_interpreter,
    _is_visible_domain_node,
    _layer_sort_key,
    _validate_node_attributions,
)
from edge2torch.utils.constants import INTERNAL_NODE_PREFIX
from edge2torch.utils.errors import Edge2TorchError


class _TinyModel(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


def test_build_feedforward_layer_interpreter_rejects_unknown_method():
    model = _TinyModel()
    layer_block = nn.Linear(2, 1)

    with pytest.raises(
        Edge2TorchError,
        match="not supported for target='nodes'",
    ):
        _build_feedforward_layer_interpreter(
            method="not_a_method",
            model=model,
            layer_block=layer_block,
            constructor_kwargs={},
        )


def test_validate_node_attributions_rejects_non_2d_tensor():
    attributions = torch.randn(2, 3, 1)

    with pytest.raises(
        Edge2TorchError,
        match="shape \\(n_examples, n_nodes\\)",
    ):
        _validate_node_attributions(
            attributions=attributions,
            layer_name="layer_1",
            sample_names=["sample_1", "sample_2"],
            node_names=["node_1", "node_2", "node_3"],
        )


def test_validate_node_attributions_rejects_wrong_row_count():
    attributions = torch.randn(3, 2)

    with pytest.raises(
        Edge2TorchError,
        match="row count.*does not match sample count",
    ):
        _validate_node_attributions(
            attributions=attributions,
            layer_name="layer_1",
            sample_names=["sample_1", "sample_2"],
            node_names=["node_1", "node_2"],
        )


def test_validate_node_attributions_rejects_wrong_width():
    attributions = torch.randn(2, 3)

    with pytest.raises(
        Edge2TorchError,
        match="width mismatch",
    ):
        _validate_node_attributions(
            attributions=attributions,
            layer_name="layer_1",
            sample_names=["sample_1", "sample_2"],
            node_names=["node_1", "node_2"],
        )


def test_validate_node_attributions_accepts_matching_shape():
    attributions = torch.randn(2, 2)

    _validate_node_attributions(
        attributions=attributions,
        layer_name="layer_1",
        sample_names=["sample_1", "sample_2"],
        node_names=["node_1", "node_2"],
    )


def test_is_visible_domain_node_returns_false_for_internal_node():
    assert _is_visible_domain_node(f"{INTERNAL_NODE_PREFIX}skip") is False


def test_is_visible_domain_node_returns_true_for_domain_node():
    assert _is_visible_domain_node("gene_a") is True


def test_layer_sort_key_returns_layer_index():
    assert _layer_sort_key("layer_12") == 12


def test_layer_sort_key_rejects_malformed_layer_name():
    with pytest.raises(Edge2TorchError, match="Invalid layer name"):
        _layer_sort_key("layer_x")


def test_layer_sort_key_rejects_missing_layer_index():
    with pytest.raises(Edge2TorchError, match="Invalid layer name"):
        _layer_sort_key("layer")
