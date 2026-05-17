"""
Expose public API functions.
"""

from importlib.metadata import PackageNotFoundError, version

from edge2torch.compile.artifact import CompileArtifact

from .align_features_to_input_nodes import align_features_to_input_nodes
from .compile_graph import compile_graph
from .customize_model import customize_model
from .interpret_model import interpret_model

try:
    __version__ = version("edge2torch")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "align_features_to_input_nodes",
    "compile_graph",
    "customize_model",
    "interpret_model",
    "CompileArtifact",
    "__version__",
]
