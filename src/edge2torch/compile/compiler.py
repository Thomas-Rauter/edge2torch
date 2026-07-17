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

from ..graph.schema import EdgeGraph
from ..utils.constants import COMPILE_BACKENDS
from ..utils.errors import Edge2TorchError
from .artifact import CompileArtifact
from .feedforward import compile_feedforward
from .state_update import compile_state_update


def compile_backend(
    graph: EdgeGraph,
    backend: str,
    bias: bool = True,
    steps: int = 3,
) -> tuple[nn.Module, CompileArtifact]:
    """
    Compile a graph into a backend-specific PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    backend
        The backend to compile to.
    bias
        Whether compiled masked linear layers include bias terms. If True,
        each target node has a learned node-level offset in addition to its
        graph-defined weighted inputs. If False, node updates are computed only
        from graph-defined weighted inputs. Disabling bias gives the graph
        structure stricter control over node activations.
    steps
        Number of state-update steps for the ``state_update`` backend.

    Returns
    -------
    tuple
        A tuple of (model, artifact).

    Raises
    ------
    Edge2TorchError
        If the backend is unsupported.
    """
    if backend == "feedforward":
        return compile_feedforward(
            graph=graph,
            bias=bias,
        )

    if backend == "state_update":
        return compile_state_update(
            graph=graph,
            bias=bias,
            steps=steps,
        )

    supported = ", ".join(sorted(COMPILE_BACKENDS))
    raise Edge2TorchError(
        f"Unsupported backend '{backend}'. Supported backends: {supported}."
    )
