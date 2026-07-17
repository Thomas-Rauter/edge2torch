import pandas as pd
import pytest

from edge2torch.compile.execution_plan import (
    build_graphnn_execution_plan,
    build_recurrent_execution_plan,
    build_state_update_execution_plan,
)
from edge2torch.graph.io import edgelist_to_graph
from edge2torch.utils.errors import Edge2TorchError


def _unreachable_output_graph():
    edgelist = pd.DataFrame(
        {
            "source": [
                "feature",
                "a",
                "b",
                "b",
            ],
            "target": [
                "prediction_good",
                "b",
                "a",
                "prediction_bad",
            ],
        }
    )

    return edgelist_to_graph(edgelist)


def test_recurrent_plan_rejects_unreachable_output():
    graph = _unreachable_output_graph()

    with pytest.raises(
        Edge2TorchError,
        match=(
            "Recurrent compilation requires every output node "
            "to be reachable from at least one input node. "
            "Unreachable output node\\(s\\): prediction_bad."
        ),
    ):
        build_recurrent_execution_plan(graph)


def test_graphnn_plan_rejects_unreachable_output():
    graph = _unreachable_output_graph()

    with pytest.raises(
        Edge2TorchError,
        match=(
            "GraphNN compilation requires every output node "
            "to be reachable from at least one input node. "
            "Unreachable output node\\(s\\): prediction_bad."
        ),
    ):
        build_graphnn_execution_plan(graph)


def _unreachable_output_relevant_graph():
    edgelist = pd.DataFrame(
        {
            "source": ["feature", "a", "b", "b"],
            "target": ["prediction", "b", "a", "prediction"],
        }
    )

    return edgelist_to_graph(edgelist)


def test_recurrent_plan_rejects_unreachable_output_relevant_nodes():
    graph = _unreachable_output_relevant_graph()

    with pytest.raises(
        Edge2TorchError,
        match="Unreachable output-relevant node\\(s\\): a, b.",
    ):
        build_recurrent_execution_plan(graph)


def test_graphnn_plan_rejects_unreachable_output_relevant_nodes():
    graph = _unreachable_output_relevant_graph()

    with pytest.raises(
        Edge2TorchError,
        match="Unreachable output-relevant node\\(s\\): a, b.",
    ):
        build_graphnn_execution_plan(graph)


def test_shared_state_update_plan_rejects_unreachable_output():
    graph = _unreachable_output_graph()

    with pytest.raises(
        Edge2TorchError,
        match=(
            "State-update compilation requires every output node "
            "to be reachable from at least one input node. "
            "Unreachable output node\\(s\\): prediction_bad."
        ),
    ):
        build_state_update_execution_plan(graph)
