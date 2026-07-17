import pandas as pd
import pytest

from edge2torch.graph.validate import (
    ValidationReport,
    _validate_common_graph_structure,
    _validate_feedforward_graph,
    _validate_state_update_backend_graph,
    _validate_state_update_graph,
    handle_validation_report,
    validate_graph,
)
from edge2torch.utils.errors import Edge2TorchError


class _Graph:
    def __init__(self, edges, nodes=None):
        self.edges = pd.DataFrame(edges, columns=["source", "target"])
        self.nodes = (
            set(self.edges["source"]).union(set(self.edges["target"]))
            if nodes is None
            else set(nodes)
        )


def test_validation_report_format_errors_without_errors():
    report = ValidationReport()

    assert report.format_errors() == "Unknown graph validation error."


def test_handle_validation_report_raises_formatted_errors():
    report = ValidationReport(errors=["first error", "second error"])

    with pytest.raises(
        Edge2TorchError,
        match="Graph validation failed",
    ):
        handle_validation_report(report=report, quiet=True)


def test_handle_validation_report_prints_notes_and_warnings(capsys):
    report = ValidationReport(
        notes=["graph note"],
        warnings=["graph warning"],
    )

    handle_validation_report(report=report, quiet=False)

    captured = capsys.readouterr()

    assert "[edge2torch] Note: graph note" in captured.out
    assert "[edge2torch] Warning: graph warning" in captured.out


def test_handle_validation_report_suppresses_notes_when_quiet(capsys):
    report = ValidationReport(
        notes=["graph note"],
        warnings=["graph warning"],
    )

    handle_validation_report(report=report, quiet=True)

    captured = capsys.readouterr()

    assert "[edge2torch] Note: graph note" not in captured.out
    assert "[edge2torch] Warning: graph warning" in captured.out


def test_validate_graph_rejects_unknown_backend():
    graph = _Graph([("a", "b")])

    with pytest.raises(Edge2TorchError, match="Unsupported backend"):
        validate_graph(graph=graph, backend="unknown")


def test_validate_common_graph_structure_reports_empty_graph():
    graph = _Graph([], nodes=[])
    report = ValidationReport()

    _validate_common_graph_structure(graph=graph, report=report)

    assert "The graph contains no edges." in report.errors
    assert "The graph contains no nodes." in report.errors


def test_validate_common_graph_structure_reports_missing_values():
    graph = _Graph([("a", None)])
    report = ValidationReport()

    _validate_common_graph_structure(graph=graph, report=report)

    assert any("missing values" in error for error in report.errors)


def test_validate_common_graph_structure_reports_empty_node_names():
    graph = _Graph([("", "b")])
    report = ValidationReport()

    _validate_common_graph_structure(graph=graph, report=report)

    assert any("empty node names" in error for error in report.errors)


def test_validate_common_graph_structure_errors_for_duplicate_edges():
    graph = _Graph(
        [
            ("a", "b"),
            ("a", "b"),
        ]
    )
    report = ValidationReport()

    _validate_common_graph_structure(graph=graph, report=report)

    assert report.errors == [
        "The graph contains 1 duplicate edge(s). "
        "Duplicate edges are not supported because edge2torch expects "
        "at most one connection for each source-target pair."
    ]
    assert report.warnings == []
    assert report.notes == ["Graph contains 2 node(s) and 2 edge(s)."]


def test_validate_feedforward_graph_rejects_terminal_outputs_at_multi_depths():
    graph = _Graph(
        [
            ("input", "early_output"),
            ("input", "hidden"),
            ("hidden", "late_output"),
        ]
    )
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any(
        "all terminal output nodes to be at the same layer depth" in error
        and "early_output" in error
        for error in report.errors
    )


def test_validate_feedforward_graph_accepts_layerable_graph():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "c"),
        ]
    )
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert not report.errors
    assert any("Feedforward backend selected" in note for note in report.notes)


def test_validate_feedforward_graph_rejects_self_loops():
    graph = _Graph([("a", "a")])
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any("self-loop" in error for error in report.errors)


def test_validate_feedforward_graph_rejects_unknown_source_node():
    graph = _Graph([("missing_source", "b")], nodes=["b"])
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any("Unknown source node" in error for error in report.errors)


def test_validate_feedforward_graph_rejects_unknown_target_node():
    graph = _Graph([("a", "missing_target")], nodes=["a"])
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any("Unknown target node" in error for error in report.errors)


def test_validate_feedforward_graph_rejects_graph_without_input_nodes():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "a"),
        ]
    )
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any("at least one input node" in error for error in report.errors)


def test_validate_feedforward_graph_rejects_unresolved_cycle():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "b"),
        ]
    )
    report = ValidationReport()

    _validate_feedforward_graph(graph=graph, report=report)

    assert any("acyclic, layerable graph" in error for error in report.errors)


def test_validate_state_update_graph_accepts_valid_graph():
    graph = _Graph(
        [
            ("input", "hidden"),
            ("hidden", "output"),
        ]
    )
    report = ValidationReport()

    _validate_state_update_backend_graph(graph=graph, report=report)

    assert not report.errors
    assert any("State-update backend selected" in note for note in report.notes)


def test_validate_state_update_graph_warns_for_self_loops():
    graph = _Graph(
        [
            ("input", "hidden"),
            ("hidden", "hidden"),
            ("hidden", "output"),
        ]
    )
    report = ValidationReport()

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="State-update",
        backend_label="state_update",
    )

    assert any("self-loop" in warning for warning in report.warnings)


def test_validate_state_update_graph_rejects_unknown_source_node():
    graph = _Graph([("missing_source", "b")], nodes=["b"])
    report = ValidationReport()

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="State-update",
        backend_label="state_update",
    )

    assert any("Unknown source node" in error for error in report.errors)


def test_validate_state_update_graph_rejects_unknown_target_node():
    graph = _Graph([("a", "missing_target")], nodes=["a"])
    report = ValidationReport()

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="State-update",
        backend_label="state_update",
    )

    assert any("Unknown target node" in error for error in report.errors)


def test_validate_state_update_graph_rejects_graph_without_input_nodes():
    graph = _Graph(
        [
            ("a", "b"),
            ("b", "a"),
            ("a", "output"),
        ]
    )
    report = ValidationReport()

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="State-update",
        backend_label="state_update",
    )

    assert any("at least one input node" in error for error in report.errors)


def test_validate_state_update_graph_rejects_graph_without_output_nodes():
    graph = _Graph(
        [
            ("input", "a"),
            ("a", "b"),
            ("b", "a"),
        ]
    )
    report = ValidationReport()

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="State-update",
        backend_label="state_update",
    )

    assert any("at least one output node" in error for error in report.errors)


@pytest.mark.parametrize(
    ("backend", "backend_name"),
    [
        ("state_update", "State-update"),
    ],
)
def test_state_update_graph_errors_for_unreachable_output(
    backend: str,
    backend_name: str,
):
    graph = _Graph(
        [
            ("feature", "prediction_good"),
            ("a", "b"),
            ("b", "a"),
            ("b", "prediction_bad"),
        ]
    )

    report = validate_graph(graph=graph, backend=backend)

    assert (
        f"{backend_name} compilation requires every output node to be "
        "reachable from at least one input node. Unreachable output "
        "node(s): prediction_bad."
    ) in report.errors


@pytest.mark.parametrize("backend", ["state_update"])
def test_state_update_graph_accepts_reachable_output_through_cycle(
    backend: str,
):
    graph = _Graph(
        [
            ("feature", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "prediction"),
        ]
    )

    report = validate_graph(graph=graph, backend=backend)

    assert report.errors == []


@pytest.mark.parametrize(
    ("backend", "backend_name"),
    [
        ("state_update", "State-update"),
    ],
)
def test_state_update_graph_reports_multiple_unreachable_outputs(
    backend: str,
    backend_name: str,
):
    graph = _Graph(
        [
            ("feature", "prediction_good"),
            ("a", "b"),
            ("b", "a"),
            ("b", "prediction_bad_a"),
            ("b", "prediction_bad_b"),
        ]
    )

    report = validate_graph(graph=graph, backend=backend)

    assert (
        f"{backend_name} compilation requires every output node to be "
        "reachable from at least one input node. Unreachable output "
        "node(s): prediction_bad_a, prediction_bad_b."
    ) in report.errors


@pytest.mark.parametrize(
    ("backend", "backend_name"),
    [
        ("state_update", "State-update"),
    ],
)
def test_state_update_graph_rejects_unreachable_output_relevant_nodes(
    backend: str,
    backend_name: str,
):
    graph = _Graph(
        [
            ("feature", "prediction"),
            ("a", "b"),
            ("b", "a"),
            ("b", "prediction"),
        ]
    )

    report = validate_graph(graph=graph, backend=backend)

    assert (
        f"{backend_name} compilation requires every node that can "
        "influence an output node to be reachable from at least one "
        "input node. Unreachable output-relevant node(s): a, b."
    ) in report.errors
