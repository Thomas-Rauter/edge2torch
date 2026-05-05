import pandas as pd
import pytest

from edge2torch.compile.execution_plan import (
    build_feedforward_execution_plan,
    build_graphnn_execution_plan,
    build_recurrent_execution_plan,
)
from edge2torch.utils.constants import PSEUDO_NODE_PREFIX
from edge2torch.utils.errors import Edge2TorchError


class _Graph:
    def __init__(self, edges, nodes=None):
        self.edges = pd.DataFrame(edges, columns=["source", "target"])
        self.nodes = (
            set(self.edges["source"]).union(set(self.edges["target"]))
            if nodes is None
            else set(nodes)
        )


# build_feedforward_execution_plan --------------------------------------------


def test_build_feedforward_execution_plan_rejects_no_input_nodes():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "a"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="at least one input node"):
        build_feedforward_execution_plan(graph)


def test_build_feedforward_execution_plan_rejects_unlayerable_cycle():
    graph = _Graph(
        [
            ("input", "a"),
            ("a", "b"),
            ("b", "a"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="cycles or cannot be layered"):
        build_feedforward_execution_plan(graph)


def test_build_feedforward_execution_plan_expands_skip_edges():
    graph = _Graph(
        [
            ("input", "hidden"),
            ("hidden", "output"),
            ("input", "output"),
        ]
    )

    plan = build_feedforward_execution_plan(graph)

    assert plan.input_node_names == ["input"]
    assert plan.output_node_names == ["output"]
    assert len(plan.pseudo_nodes) == 1
    assert plan.pseudo_nodes[0].startswith(PSEUDO_NODE_PREFIX)

    pseudo_node = plan.pseudo_nodes[0]

    assert pseudo_node in plan.node_names_by_layer["layer_1"]
    assert plan.node_to_layer[pseudo_node] == "layer_1"

    expanded_edges = set(
        zip(
            plan.expanded_edges["source"],
            plan.expanded_edges["target"],
        )
    )

    assert ("input", "hidden") in expanded_edges
    assert ("hidden", "output") in expanded_edges
    assert ("input", pseudo_node) in expanded_edges
    assert (pseudo_node, "output") in expanded_edges


def test_build_feedforward_execution_plan_keeps_adjacent_edges():
    graph = _Graph(
        [
            ("gene_a", "output"),
            ("gene_b", "output"),
        ]
    )

    plan = build_feedforward_execution_plan(graph)

    expected_edges = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b"],
            "target": ["output", "output"],
        }
    )

    pd.testing.assert_frame_equal(
        plan.expanded_edges.sort_values(["source", "target"]).reset_index(
            drop=True
        ),
        expected_edges,
    )

    assert plan.pseudo_nodes == []
    assert plan.node_names_by_layer == {
        "layer_0": ["gene_a", "gene_b"],
        "layer_1": ["output"],
    }
    assert plan.node_to_layer == {
        "gene_a": "layer_0",
        "gene_b": "layer_0",
        "output": "layer_1",
    }


# build_recurrent_execution_plan ----------------------------------------------


def test_build_recurrent_execution_plan_rejects_empty_edges():
    graph = _Graph([], nodes=["input"])

    with pytest.raises(Edge2TorchError, match="at least one edge"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_rejects_empty_nodes():
    graph = _Graph([("input", "output")], nodes=[])

    with pytest.raises(Edge2TorchError, match="at least one node"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_rejects_unknown_source_node():
    graph = _Graph(
        [("missing_source", "output")],
        nodes=["output"],
    )

    with pytest.raises(Edge2TorchError, match="Unknown source node"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_rejects_unknown_target_node():
    graph = _Graph(
        [("input", "missing_target")],
        nodes=["input"],
    )

    with pytest.raises(Edge2TorchError, match="Unknown target node"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_rejects_no_input_nodes():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "a"),
            ("a", "output"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="at least one input node"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_rejects_no_output_nodes():
    graph = _Graph(
        [
            ("input", "a"),
            ("a", "b"),
            ("b", "a"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="at least one output node"):
        build_recurrent_execution_plan(graph)


def test_build_recurrent_execution_plan_returns_sorted_names():
    graph = _Graph(
        [
            ("input_b", "hidden"),
            ("input_a", "hidden"),
            ("hidden", "output_b"),
            ("hidden", "output_a"),
        ]
    )

    plan = build_recurrent_execution_plan(graph)

    assert plan.node_names == [
        "hidden",
        "input_a",
        "input_b",
        "output_a",
        "output_b",
    ]
    assert plan.input_node_names == ["input_a", "input_b"]
    assert plan.output_node_names == ["output_a", "output_b"]


# build_graphnn_execution_plan -------------------------------------------------


def test_build_graphnn_execution_plan_rejects_empty_edges():
    graph = _Graph([], nodes=["input"])

    with pytest.raises(Edge2TorchError, match="at least one edge"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_rejects_empty_nodes():
    graph = _Graph([("input", "output")], nodes=[])

    with pytest.raises(Edge2TorchError, match="at least one node"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_rejects_unknown_source_node():
    graph = _Graph(
        [("missing_source", "output")],
        nodes=["output"],
    )

    with pytest.raises(Edge2TorchError, match="Unknown source node"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_rejects_unknown_target_node():
    graph = _Graph(
        [("input", "missing_target")],
        nodes=["input"],
    )

    with pytest.raises(Edge2TorchError, match="Unknown target node"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_rejects_no_input_nodes():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "a"),
            ("a", "output"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="at least one input node"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_rejects_no_output_nodes():
    graph = _Graph(
        [
            ("input", "a"),
            ("a", "b"),
            ("b", "a"),
        ]
    )

    with pytest.raises(Edge2TorchError, match="at least one output node"):
        build_graphnn_execution_plan(graph)


def test_build_graphnn_execution_plan_returns_sorted_names():
    graph = _Graph(
        [
            ("input_b", "hidden"),
            ("input_a", "hidden"),
            ("hidden", "output_b"),
            ("hidden", "output_a"),
        ]
    )

    plan = build_graphnn_execution_plan(graph)

    assert plan.node_names == [
        "hidden",
        "input_a",
        "input_b",
        "output_a",
        "output_b",
    ]
    assert plan.input_node_names == ["input_a", "input_b"]
    assert plan.output_node_names == ["output_a", "output_b"]
