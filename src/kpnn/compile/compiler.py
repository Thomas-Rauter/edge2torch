from .feedforward import compile_feedforward
from .recurrent import compile_recurrent
# from .graph import compile_graph
from ..utils.errors import KPNNError


def compile_backend(
    graph,
    backend,
):
    """
    Compile a graph into a backend-specific PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    backend
        Backend to compile to.

    Returns
    -------
    tuple
        A tuple of (model, artifact).

    Raises
    ------
    KPNNError
        If the backend is unsupported.
    """
    if backend == "feedforward":
        return compile_feedforward(graph)

    if backend == "recurrent":
        return compile_recurrent(graph)

    # if backend == "graph":
        # return compile_graph(graph)

    raise KPNNError(f"Unsupported backend '{backend}'.")
