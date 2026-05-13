import pandas as pd
import pytest
import torch

from edge2torch.nn.model import (
    EdgeGraphNNModel,
    EdgeModel,
    RecurrentEdgeModel,
)
from edge2torch.utils.errors import Edge2TorchError


class _FeedforwardPlan:
    def __init__(
        self,
        *,
        node_names_by_layer=None,
        expanded_edges=None,
    ):
        self.node_names_by_layer = (
            {
                "layer_0": ["gene_a", "gene_b"],
                "layer_1": ["output_1"],
            }
            if node_names_by_layer is None
            else node_names_by_layer
        )
        self.expanded_edges = (
            pd.DataFrame(
                {
                    "source": ["gene_a", "gene_b"],
                    "target": ["output_1", "output_1"],
                }
            )
            if expanded_edges is None
            else expanded_edges
        )


class _RecurrentPlan:
    def __init__(
        self,
        *,
        node_names=None,
        input_node_names=None,
        output_node_names=None,
        original_edges=None,
    ):
        self.node_names = (
            ["input_1", "hidden_1", "output_1"]
            if node_names is None
            else node_names
        )
        self.input_node_names = (
            ["input_1"] if input_node_names is None else input_node_names
        )
        self.output_node_names = (
            ["output_1"] if output_node_names is None else output_node_names
        )
        self.original_edges = (
            pd.DataFrame(
                {
                    "source": ["input_1", "hidden_1"],
                    "target": ["hidden_1", "output_1"],
                }
            )
            if original_edges is None
            else original_edges
        )


class _GraphNNPlan:
    def __init__(
        self,
        *,
        node_names=None,
        input_node_names=None,
        output_node_names=None,
        original_edges=None,
    ):
        self.node_names = (
            ["input_1", "hidden_1", "output_1"]
            if node_names is None
            else node_names
        )
        self.input_node_names = (
            ["input_1"] if input_node_names is None else input_node_names
        )
        self.output_node_names = (
            ["output_1"] if output_node_names is None else output_node_names
        )
        self.original_edges = (
            pd.DataFrame(
                {
                    "source": ["input_1", "hidden_1"],
                    "target": ["hidden_1", "output_1"],
                }
            )
            if original_edges is None
            else original_edges
        )


# EdgeModel --------------------------------------------------------------------


def test_edge_model_get_layer_block_rejects_invalid_layer_prefix():
    model = EdgeModel(_FeedforwardPlan())

    with pytest.raises(Edge2TorchError, match="Invalid layer name"):
        model._edge2torch_get_feedforward_layer_block("hidden_1")


def test_edge_model_get_layer_block_rejects_malformed_layer_name():
    model = EdgeModel(_FeedforwardPlan())

    with pytest.raises(Edge2TorchError, match="Invalid layer name"):
        model._edge2torch_get_feedforward_layer_block("layer_x")


def test_edge_model_get_layer_block_rejects_input_layer():
    model = EdgeModel(_FeedforwardPlan())

    with pytest.raises(Edge2TorchError, match="input layer"):
        model._edge2torch_get_feedforward_layer_block("layer_0")


def test_edge_model_get_layer_block_rejects_unknown_layer_name():
    model = EdgeModel(_FeedforwardPlan())

    with pytest.raises(Edge2TorchError, match="Unknown layer name"):
        model._edge2torch_get_feedforward_layer_block("layer_2")


def test_edge_model_get_layer_block_rejects_missing_block_for_known_layer():
    model = EdgeModel(_FeedforwardPlan())
    model.layer_names.append("layer_2")

    with pytest.raises(Edge2TorchError, match="No block exists"):
        model._edge2torch_get_feedforward_layer_block("layer_2")


def test_edge_model_sort_layer_names_rejects_invalid_layer_name():
    with pytest.raises(Edge2TorchError, match="Invalid layer name"):
        EdgeModel._sort_layer_names(["layer_0", "bad_layer"])


def test_edge_model_select_block_edges_filters_edges_between_layers():
    expanded_edges = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b", "other"],
            "target": ["hidden_1", "hidden_1", "hidden_1"],
        }
    )

    result = EdgeModel._select_block_edges(
        expanded_edges=expanded_edges,
        input_node_names=["gene_a", "gene_b"],
        output_node_names=["hidden_1"],
    )

    expected = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b"],
            "target": ["hidden_1", "hidden_1"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


# RecurrentEdgeModel -----------------------------------------------------------


def test_recurrent_edge_model_rejects_non_positive_steps():
    with pytest.raises(Edge2TorchError, match="positive integer"):
        RecurrentEdgeModel(
            execution_plan=_RecurrentPlan(),
            steps=0,
        )


def test_recurrent_edge_model_rejects_non_integer_steps():
    with pytest.raises(Edge2TorchError, match="positive integer"):
        RecurrentEdgeModel(
            execution_plan=_RecurrentPlan(),
            steps=1.5,
        )


def test_recurrent_edge_model_rejects_missing_input_nodes():
    plan = _RecurrentPlan(input_node_names=[])

    with pytest.raises(Edge2TorchError, match="at least one input node"):
        RecurrentEdgeModel(execution_plan=plan)


def test_recurrent_edge_model_rejects_missing_output_nodes():
    plan = _RecurrentPlan(output_node_names=[])

    with pytest.raises(Edge2TorchError, match="at least one output node"):
        RecurrentEdgeModel(execution_plan=plan)


def test_recurrent_edge_model_forward_rejects_1d_input():
    model = RecurrentEdgeModel(execution_plan=_RecurrentPlan())

    with pytest.raises(Edge2TorchError, match="2-dimensional"):
        model(torch.tensor([1.0]))


def test_recurrent_edge_model_forward_rejects_wrong_input_width():
    model = RecurrentEdgeModel(execution_plan=_RecurrentPlan())

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        model(torch.randn(2, 2))


def test_recurrent_edge_model_forward_returns_output_nodes():
    model = RecurrentEdgeModel(
        execution_plan=_RecurrentPlan(),
        steps=2,
        bias=False,
    )

    x = torch.tensor([[1.0], [2.0]])

    result = model(x)

    assert result.shape == (2, 1)


# EdgeGraphNNModel -------------------------------------------------------------


def test_edge_graphnn_model_rejects_non_positive_steps():
    with pytest.raises(Edge2TorchError, match="positive integer"):
        EdgeGraphNNModel(
            execution_plan=_GraphNNPlan(),
            steps=0,
        )


def test_edge_graphnn_model_rejects_non_integer_steps():
    with pytest.raises(Edge2TorchError, match="positive integer"):
        EdgeGraphNNModel(
            execution_plan=_GraphNNPlan(),
            steps=1.5,
        )


def test_edge_graphnn_model_rejects_missing_input_nodes():
    plan = _GraphNNPlan(input_node_names=[])

    with pytest.raises(Edge2TorchError, match="at least one input node"):
        EdgeGraphNNModel(execution_plan=plan)


def test_edge_graphnn_model_rejects_missing_output_nodes():
    plan = _GraphNNPlan(output_node_names=[])

    with pytest.raises(Edge2TorchError, match="at least one output node"):
        EdgeGraphNNModel(execution_plan=plan)


def test_edge_graphnn_model_forward_rejects_1d_input():
    model = EdgeGraphNNModel(execution_plan=_GraphNNPlan())

    with pytest.raises(Edge2TorchError, match="2-dimensional"):
        model(torch.tensor([1.0]))


def test_edge_graphnn_model_forward_rejects_wrong_input_width():
    model = EdgeGraphNNModel(execution_plan=_GraphNNPlan())

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        model(torch.randn(2, 2))


def test_edge_graphnn_model_forward_returns_output_nodes():
    model = EdgeGraphNNModel(
        execution_plan=_GraphNNPlan(),
        steps=2,
        bias=False,
    )

    x = torch.tensor([[1.0], [2.0]])

    result = model(x)

    assert result.shape == (2, 1)
