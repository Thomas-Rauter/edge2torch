import pandas as pd

from .compile.compiler import compile_backend
from .compile.input_validation import validate_compile_graph_inputs
from .graph.io import edgelist_to_graph
from .graph.validate import handle_validation_report, validate_graph


def compile_graph(
    edgelist: pd.DataFrame,
    backend: str = "feedforward",
    quiet: bool = False,
    bias: bool = True,
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

    The edgelist may optionally include edge-level parameter metadata using the
    columns ``"initial_weight"`` and ``"constraint"``. These columns allow
    individual edges to define their initial effective weight and, where
    supported by the selected backend, constrain the trainable edge weight
    during optimization. Supported constraint values are ``"unconstrained"``,
    ``"positive"``, ``"negative"``, and ``"fixed"``. If omitted, edges use the
    backend's default trainable weight initialization and unconstrained weight
    behavior.

    Graph-derived connectivity is enforced through masks on trainable edge
    weights. By default, compiled layers also include bias terms. Biases are
    node-level parameters, not graph edges, and are not constrained by the edge
    mask. Set ``bias=False`` to remove these offsets so node updates depend only
    on graph-defined weighted inputs.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``"source"`` and ``"target"``.
        Each row defines a directed connection from one named node to another,
        following the direction of computation. The table must include edges
        from input feature nodes into the rest of the architecture graph.

        The table may also include optional columns ``"initial_weight"`` and
        ``"constraint"``. If provided, ``"initial_weight"`` defines the initial
        effective edge weight, and ``"constraint"`` defines how that edge weight
        is parameterized during training. Supported constraints are:

        - ``"unconstrained"``: the edge weight is trainable and may become
          positive or negative.
        - ``"positive"``: the edge weight is trainable and constrained to remain
          positive.
        - ``"negative"``: the edge weight is trainable and constrained to remain
          negative.
        - ``"fixed"``: the edge weight is fixed to ``"initial_weight"`` and is
          not trainable.

        The optional columns are independent and row-wise sparse. Missing
        ``"initial_weight"`` values use the backend's default initialization
        for that edge. Missing ``"constraint"`` values are treated as
        ``"unconstrained"`` for that edge.

        If both values are provided for an edge, positive-constrained edges
        must have positive initial weights and negative-constrained edges must
        have negative initial weights. Edges with ``constraint="fixed"`` must
        provide an ``"initial_weight"`` value in the same row, because fixed
        edges require an explicit constant value.
    backend : str, default="feedforward"
        Backend to compile to. One of ``"feedforward"``, ``"recurrent"``,
        or ``"graphnn"``.
    quiet : bool, default=False
        If False, emit informational notes during validation. If True,
        suppress informational notes.
    bias : bool, default=True
        Whether compiled masked linear layers include bias terms. If True,
        each target node has a learned node-level offset in addition to its
        graph-defined weighted inputs. If False, node updates are computed only
        from graph-defined weighted inputs. Disabling bias gives the graph
        structure stricter control over node activations.

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

    Compile a recurrent architecture with edge-level initial weights and
    constraints.

    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["feature_a", "feature_b", "hidden_1"],
    ...         "target": ["hidden_1", "hidden_1", "prediction"],
    ...         "initial_weight": [0.1, -0.2, 0.5],
    ...         "constraint": ["positive", "negative", "fixed"],
    ...     }
    ... )
    >>>
    >>> model, artifact = compile_graph(
    ...     edgelist=edgelist,
    ...     backend="recurrent",
    ...     quiet=True,
    ... )
    """
    validate_compile_graph_inputs(
        edgelist=edgelist,
        backend=backend,
        quiet=quiet,
        bias=bias,
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
        bias=bias,
    )

    return model, artifact
