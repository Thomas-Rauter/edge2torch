"""Tests that recurrent and graphnn share one state-update core."""

from pathlib import Path

import pandas as pd
import torch

from edge2torch.compile.execution_plan import (
    GraphNNExecutionPlan,
    RecurrentExecutionPlan,
    StateUpdateExecutionPlan,
    build_graphnn_execution_plan,
    build_recurrent_execution_plan,
    build_state_update_execution_plan,
)
from edge2torch.compile_graph import compile_graph
from edge2torch.graph.io import edgelist_to_graph
from edge2torch.nn.masked_linear import ConstrainedMaskedLinear
from edge2torch.nn.model import (
    EdgeGraphNNModel,
    RecurrentEdgeModel,
    StateUpdateEdgeModel,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "edgelists"


def test_state_update_plan_aliases_are_the_same_type():
    assert RecurrentExecutionPlan is StateUpdateExecutionPlan
    assert GraphNNExecutionPlan is StateUpdateExecutionPlan


def test_backend_plan_wrappers_match_shared_builder():
    edgelist = pd.read_csv(FIXTURES / "recurrent_cycle.csv")
    graph = edgelist_to_graph(edgelist)

    shared = build_state_update_execution_plan(graph)
    recurrent = build_recurrent_execution_plan(graph)
    graphnn = build_graphnn_execution_plan(graph)

    assert isinstance(shared, StateUpdateExecutionPlan)
    assert recurrent.node_names == shared.node_names
    assert graphnn.node_names == shared.node_names
    assert recurrent.input_node_names == shared.input_node_names
    assert graphnn.input_node_names == shared.input_node_names
    assert recurrent.output_node_names == shared.output_node_names
    assert graphnn.output_node_names == shared.output_node_names
    pd.testing.assert_frame_equal(
        recurrent.original_edges,
        shared.original_edges,
    )
    pd.testing.assert_frame_equal(
        graphnn.original_edges,
        shared.original_edges,
    )


def test_backend_models_are_state_update_subclasses():
    assert issubclass(RecurrentEdgeModel, StateUpdateEdgeModel)
    assert issubclass(EdgeGraphNNModel, StateUpdateEdgeModel)


def test_recurrent_and_graphnn_match_with_synced_weights():
    edgelist = pd.read_csv(FIXTURES / "recurrent_cycle.csv")

    recurrent_model, recurrent_artifact = compile_graph(
        edgelist,
        backend="recurrent",
        steps=3,
        bias=True,
        quiet=True,
    )
    graphnn_model, graphnn_artifact = compile_graph(
        edgelist,
        backend="graphnn",
        steps=3,
        bias=True,
        quiet=True,
    )

    assert isinstance(recurrent_model, StateUpdateEdgeModel)
    assert isinstance(graphnn_model, StateUpdateEdgeModel)
    assert recurrent_artifact.backend == "recurrent"
    assert graphnn_artifact.backend == "graphnn"
    assert isinstance(
        recurrent_artifact.execution_plan,
        StateUpdateExecutionPlan,
    )
    assert isinstance(
        graphnn_artifact.execution_plan,
        StateUpdateExecutionPlan,
    )

    with torch.no_grad():
        if hasattr(recurrent_model.state_linear, "weight"):
            graphnn_model.state_linear.weight.copy_(
                recurrent_model.state_linear.weight
            )
        elif isinstance(
            recurrent_model.state_linear,
            ConstrainedMaskedLinear,
        ) and isinstance(
            graphnn_model.state_linear,
            ConstrainedMaskedLinear,
        ):
            graphnn_model.state_linear.raw_weight.copy_(
                recurrent_model.state_linear.raw_weight
            )

        if (
            recurrent_model.state_linear.bias is not None
            and graphnn_model.state_linear.bias is not None
        ):
            graphnn_model.state_linear.bias.copy_(
                recurrent_model.state_linear.bias
            )

    x = torch.randn(5, len(recurrent_artifact.feature_names))

    torch.testing.assert_close(
        recurrent_model(x),
        graphnn_model(x),
    )


def test_legacy_linear_aliases_point_to_state_linear():
    edgelist = pd.DataFrame(
        {
            "source": ["input_1", "hidden_1"],
            "target": ["hidden_1", "output_1"],
        }
    )

    recurrent_model, _ = compile_graph(
        edgelist,
        backend="recurrent",
        quiet=True,
    )
    graphnn_model, _ = compile_graph(
        edgelist,
        backend="graphnn",
        quiet=True,
    )

    assert recurrent_model.recurrent is recurrent_model.state_linear
    assert graphnn_model.message_passing is graphnn_model.state_linear
    parameter_names = {name for name, _ in recurrent_model.named_parameters()}
    assert "recurrent.weight" in parameter_names
    assert "state_linear.weight" not in parameter_names
