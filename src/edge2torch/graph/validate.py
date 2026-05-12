"""
Internal graph validation and reporting.

Why this file exists
--------------------
This file separates graph-structural validation from both public API
input validation and backend-specific compilation. The package needs a
place to check whether a normalized internal graph is structurally
suitable for a requested backend and to represent the result in a form
that can be handled consistently.

Role in the package
-------------------
This is an internal graph-validation module. It defines the validation
report structure, performs backend-aware validation on normalized graph
objects, and handles the resulting notes, warnings, and errors. It
should focus on graph validity and reporting, not on input conversion,
public API orchestration, or model construction.
"""

from dataclasses import dataclass, field

from ..utils.errors import Edge2TorchError
from .schema import EdgeGraph


@dataclass
class ValidationReport:
    """
    Container for graph validation results.
    """

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """
        Return whether the report contains any errors.
        """
        return len(self.errors) > 0

    def format_errors(self) -> str:
        """
        Format all errors into a single readable message.

        Returns
        -------
        str
            Combined error message.
        """
        if not self.errors:
            return "Unknown graph validation error."

        header = "Graph validation failed with the following errors:"
        body = "\n".join(f"- {error}" for error in self.errors)
        return f"{header}\n{body}"


# Level 1 function definitions (functions called by API functions) -------------


def validate_graph(
    graph: EdgeGraph,
    backend: str,
) -> ValidationReport:
    """
    Validate a normalized internal graph for backend-specific compilation.

    Parameters
    ----------
    graph : EdgeGraph
        Internal graph object.
    backend : str
        Backend to compile to.

    Returns
    -------
    ValidationReport
        Validation report containing notes, warnings, and errors.
    """
    report = ValidationReport()

    _validate_common_graph_structure(
        graph=graph,
        report=report,
    )

    if backend == "feedforward":
        _validate_feedforward_graph(
            graph=graph,
            report=report,
        )
    elif backend == "recurrent":
        _validate_recurrent_graph(
            graph=graph,
            report=report,
        )
    elif backend == "graphnn":
        _validate_graphnn_graph(
            graph=graph,
            report=report,
        )
    else:
        raise Edge2TorchError(
            f"Unsupported backend '{backend}' during graph validation."
        )

    return report


def handle_validation_report(
    report: ValidationReport,
    quiet: bool,
) -> None:
    """
    Handle a validation report by raising errors and printing notes or warnings.

    Parameters
    ----------
    report : ValidationReport
        Validation report to handle.
    quiet : bool
        If True, suppress notes. Warnings are still shown.

    Raises
    ------
    Edge2TorchError
        If the report contains errors.
    """
    if report.has_errors:
        raise Edge2TorchError(report.format_errors())

    if not quiet:
        for note in report.notes:
            print(f"[edge2torch] Note: {note}")

    for warning in report.warnings:
        print(f"[edge2torch] Warning: {warning}")


# Level 2 function definitions (functions called by level 1 functions) ---------


def _validate_common_graph_structure(
    graph: EdgeGraph,
    report: ValidationReport,
) -> None:
    """
    Validate backend-independent graph structure.
    """
    if graph.edges.empty:
        report.errors.append("The graph contains no edges.")

    if len(graph.nodes) == 0:
        report.errors.append("The graph contains no nodes.")

    if graph.edges[["source", "target"]].isnull().any().any():
        report.errors.append(
            "The graph contains missing values in 'source' or 'target'."
        )

    empty_source = graph.edges["source"] == ""
    empty_target = graph.edges["target"] == ""

    if empty_source.any() or empty_target.any():
        report.errors.append(
            "The graph contains empty node names in 'source' or 'target'."
        )

    duplicate_edges = graph.edges.duplicated(subset=["source", "target"])
    n_duplicate_edges = int(duplicate_edges.sum())

    if n_duplicate_edges > 0:
        report.errors.append(
            f"The graph contains {n_duplicate_edges} duplicate edge(s). "
            "Duplicate edges are not supported because edge2torch treats the "
            "edgelist as a binary connectivity graph."
        )

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    report.notes.append(
        f"Graph contains {n_nodes} node(s) and {n_edges} edge(s)."
    )


def _validate_feedforward_graph(
    graph: EdgeGraph,
    report: ValidationReport,
) -> None:
    """
    Validate graph constraints required by the feedforward backend.
    """
    report.notes.append(
        "Feedforward backend selected. Graph must be layerable."
    )

    self_loops = graph.edges["source"] == graph.edges["target"]
    n_self_loops = int(self_loops.sum())

    if n_self_loops > 0:
        report.errors.append(
            f"The graph contains {n_self_loops} self-loop edge(s). "
            "Self-loops are not allowed for the feedforward backend."
        )

    if report.has_errors:
        return

    node_names = list(graph.nodes)

    in_degree: dict[str, int] = {node: 0 for node in node_names}
    children: dict[str, list[str]] = {node: [] for node in node_names}

    for row in graph.edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        if source not in children:
            report.errors.append(
                f"Unknown source node '{source}' in feedforward graph."
            )
            continue

        if target not in in_degree:
            report.errors.append(
                f"Unknown target node '{target}' in feedforward graph."
            )
            continue

        children[source].append(target)
        in_degree[target] += 1

    if report.has_errors:
        return

    current_layer_nodes = sorted(
        node for node in node_names if in_degree[node] == 0
    )

    if not current_layer_nodes:
        report.errors.append(
            "Feedforward compilation requires at least one input node."
        )
        return

    visited_nodes: set[str] = set()

    while current_layer_nodes:
        next_layer_candidates: set[str] = set()

        for node in current_layer_nodes:
            visited_nodes.add(node)

            for child in children[node]:
                in_degree[child] -= 1

                if in_degree[child] == 0:
                    next_layer_candidates.add(child)

        current_layer_nodes = sorted(next_layer_candidates)

    if len(visited_nodes) != len(node_names):
        report.errors.append(
            "Feedforward compilation requires an acyclic, layerable graph. "
            "The graph contains at least one cycle or unresolved dependency."
        )


def _validate_recurrent_graph(
    graph: EdgeGraph,
    report: ValidationReport,
) -> None:
    """
    Validate graph constraints required by the recurrent backend.
    """
    report.notes.append("Recurrent backend selected. Cycles may be allowed.")

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="Recurrent",
        backend_label="recurrent",
    )


def _validate_graphnn_graph(
    graph: EdgeGraph,
    report: ValidationReport,
) -> None:
    """
    Validate graph constraints required by the graphnn backend.
    """
    report.notes.append(
        "GraphNN backend selected. Graph structure will be compiled "
        "for message passing."
    )

    _validate_state_update_graph(
        graph=graph,
        report=report,
        backend_name="GraphNN",
        backend_label="graphnn",
    )


# Level 3 functions (functions called by level 2 functions) --------------------


def _validate_state_update_graph(
    graph: EdgeGraph,
    report: ValidationReport,
    backend_name: str,
    backend_label: str,
) -> None:
    """
    Validate graph constraints shared by state-update backends.

    Recurrent and graphnn backends both allow cycles, but they still require
    at least one input node and at least one output node. Input nodes are
    inferred as nodes with no incoming edges. Output nodes are inferred as
    nodes with no outgoing edges.
    """
    self_loops = graph.edges["source"] == graph.edges["target"]
    n_self_loops = int(self_loops.sum())

    if n_self_loops > 0:
        report.warnings.append(
            f"The graph contains {n_self_loops} self-loop edge(s)."
        )

    node_names = list(graph.nodes)

    children: dict[str, list[str]] = {node: [] for node in node_names}
    parents: dict[str, list[str]] = {node: [] for node in node_names}

    for row in graph.edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        if source not in children:
            report.errors.append(
                f"Unknown source node '{source}' in {backend_label} graph."
            )
            continue

        if target not in parents:
            report.errors.append(
                f"Unknown target node '{target}' in {backend_label} graph."
            )
            continue

        children[source].append(target)
        parents[target].append(source)

    if report.has_errors:
        return

    input_node_names = sorted(
        node for node in node_names if len(parents[node]) == 0
    )

    output_node_names = sorted(
        node for node in node_names if len(children[node]) == 0
    )

    if not input_node_names:
        report.errors.append(
            f"{backend_name} compilation requires at least one input node. "
            "Cycles are allowed, but the graph must include at least one "
            "node with no incoming edges."
        )

    if not output_node_names:
        report.errors.append(
            f"{backend_name} compilation requires at least one output node. "
            "Cycles are allowed, but the graph must include at least one "
            "node with no outgoing edges."
        )
