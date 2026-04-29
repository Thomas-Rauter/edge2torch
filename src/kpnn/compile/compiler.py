"""
Backend dispatch for graph compilation.

Why this file exists
--------------------
This file centralizes the mapping from backend names to backend-specific
compiler implementations. Keeping this dispatch logic in one place makes
the compilation architecture easier to extend and avoids scattering
backend selection rules across the package.

Role in the package
-------------------
This is an internal compilation orchestration module. It chooses which
backend-specific compiler should be called for a validated graph and
backend name. It should contain backend dispatch logic, not the backend
implementations themselves or the public API entry point.
"""

from torch import nn

from ..graph.schema import KPNNGraph
from ..utils.errors import KPNNError
from .artifact import KPNNArtifact
from .feedforward import compile_feedforward
from .graphnn import compile_graphnn
from .recurrent import compile_recurrent


def compile_backend(
    graph: KPNNGraph,
    backend: str,
) -> tuple[nn.Module, KPNNArtifact]:
    """
    Compile a graph into a backend-specific PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    backend
        The backend to compile to.

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

    if backend == "graphnn":
        return compile_graphnn(graph)

    raise KPNNError(f"Unsupported backend '{backend}'.")
