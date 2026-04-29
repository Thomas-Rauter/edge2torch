"""
API function
"""

import pandas as pd

from .compile.compiler import compile_backend
from .compile.input_validation import validate_compile_graph_inputs
from .graph.io import edgelist_to_graph
from .graph.validate import handle_validation_report, validate_graph


def compile_graph(
    edgelist: pd.DataFrame,
    backend: str = "feedforward",
    quiet: bool = False,
):
    """
    Compile an edgelist into a PyTorch model and compilation artifact.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns 'source' and 'target'.
    backend : str, default="feedforward"
        Backend to compile to. One of: "feedforward", "recurrent", "graphnn".
    quiet : bool, default=False
        If False, emit informational notes during validation. If True,
        suppress notes and only surface warnings and errors.

    Returns
    -------
    tuple
        A tuple of (model, artifact).

    Raises
    ------
    ValueError
        If the backend is unknown.
    KPNNError
        If graph validation fails.
    """
    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend=backend,
        quiet=quiet,
    )

    graph = edgelist_to_graph(edgelist)

    report = validate_graph(
        graph=graph,
        backend=backend,
    )

    handle_validation_report(
        report=report,
        quiet=quiet,
    )

    model, artifact = compile_backend(
        graph=graph,
        backend=backend,
    )

    return model, artifact
