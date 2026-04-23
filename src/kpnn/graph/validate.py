from dataclasses import dataclass, field

from ..utils.errors import KPNNError
from .schema import KPNNGraph


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


def validate_graph(
    graph: KPNNGraph,
    backend: str,
) -> ValidationReport:
    """
    Validate a normalized internal graph for compilation.

    Parameters
    ----------
    graph : KPNNGraph
        Internal graph object.
    backend : str
        Backend to compile to.

    Returns
    -------
    ValidationReport
        Validation report containing notes, warnings, and errors.
    """
    report = ValidationReport()

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

    self_loops = graph.edges["source"] == graph.edges["target"]
    n_self_loops = int(self_loops.sum())

    if n_self_loops > 0:
        report.warnings.append(
            f"The graph contains {n_self_loops} self-loop edge(s)."
        )

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    report.notes.append(
        f"Graph contains {n_nodes} node(s) and {n_edges} edge(s)."
    )

    if backend == "feedforward":
        report.notes.append(
            "Feedforward backend selected. Graph must be layerable."
        )
    elif backend == "recurrent":
        report.notes.append(
            "Recurrent backend selected. Cycles may be allowed."
        )
    elif backend == "graphnn":
        report.notes.append(
            "GraphNN backend selected. Graph structure will be compiled "
            "for message passing."
        )
    else:
        raise KPNNError(
            f"Unsupported backend '{backend}' during graph validation."
        )

    return report


def handle_validation_report(
    report: ValidationReport,
    quiet: bool,
) -> None:
    """
    Handle a validation report by raising errors and printing messages.

    Parameters
    ----------
    report : ValidationReport
        Validation report to handle.
    quiet : bool
        If True, suppress notes. Warnings are still shown.

    Raises
    ------
    KPNNError
        If the report contains errors.
    """
    if report.has_errors:
        raise KPNNError(report.format_errors())

    if not quiet:
        for note in report.notes:
            print(f"[kpnn] Note: {note}")

    for warning in report.warnings:
        print(f"[kpnn] Warning: {warning}")
