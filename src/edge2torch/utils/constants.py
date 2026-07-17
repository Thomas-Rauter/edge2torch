"""
Shared internal constants for the edge2torch package.
"""

from typing import Literal, TypeAlias

INTERNAL_NODE_PREFIX = "__edge2torch_"
PSEUDO_NODE_PREFIX = "__edge2torch_pseudo__"

CompileBackend: TypeAlias = Literal["feedforward", "state_update"]
COMPILE_BACKENDS: frozenset[CompileBackend] = frozenset(
    ("feedforward", "state_update")
)
