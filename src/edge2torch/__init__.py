"""
Expose public API functions.
"""

from importlib.metadata import PackageNotFoundError, version

from edge2torch.compile.artifact import CompileArtifact

from .align_features_to_input_nodes import align_features_to_input_nodes
from .compile_graph import compile_graph
from .customize_model import customize_model
from .graph_topology import GraphTopology, graph_topology
from .interpret_model import interpret_model
from .utils.constants import COMPILE_BACKENDS, CompileBackend

try:
    __version__ = version("edge2torch")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "align_features_to_input_nodes",
    "compile_graph",
    "customize_model",
    "graph_topology",
    "interpret_model",
    "CompileArtifact",
    "CompileBackend",
    "COMPILE_BACKENDS",
    "GraphTopology",
    "__version__",
]
