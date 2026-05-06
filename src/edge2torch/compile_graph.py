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
    Compile an edgelist into a sparse PyTorch model and compilation artifact.

    The edgelist defines the architecture graph. Each row describes one
    directed connection from ``source`` to ``target`` in the direction of
    computation. In other words, edges should point from input feature nodes
    toward hidden nodes and output nodes.

    Input features are inferred as graph nodes with no incoming edges. Output
    nodes are inferred as graph nodes with no outgoing edges. The returned
    artifact stores the inferred input feature names in
    ``artifact.feature_names``. Tensors passed to the compiled model must have
    columns in that exact order.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``"source"`` and ``"target"``.
        Each row defines a directed connection from one named node to another,
        following the direction of computation. The table must include edges
        from input feature nodes into the rest of the architecture graph.
    backend : str, default="feedforward"
        Backend to compile to. One of ``"feedforward"``, ``"recurrent"``,
        or ``"graphnn"``.
    quiet : bool, default=False
        If False, emit informational notes during validation. If True,
        suppress informational notes.

    Returns
    -------
    tuple[torch.nn.Module, CompileArtifact]
        Tuple ``(model, artifact)``. ``model`` is a PyTorch ``nn.Module``
        compiled from the edgelist. ``artifact`` stores compilation metadata,
        including ``artifact.feature_names``.

    Raises
    ------
    Edge2TorchError
        If input validation, graph validation, or backend compilation fails.

    Examples
    --------
    Compile a small feedforward architecture from an edgelist.

    >>> import pandas as pd
    >>> from edge2torch import compile_graph
    >>>
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["feature_a", "feature_b", "hidden_1"],
    ...         "target": ["hidden_1", "hidden_1", "prediction"],
    ...     }
    ... )
    >>>
    >>> model, artifact = compile_graph(
    ...     edgelist=edgelist,
    ...     backend="feedforward",
    ...     quiet=True,
    ... )
    >>>
    >>> artifact.feature_names
    ['feature_a', 'feature_b']
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
