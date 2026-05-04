"""
Validation logic for the compile_graph() public API.

Why this file exists
--------------------
This file separates strict public input validation from both the
compile_graph() API wrapper and the internal compiler implementation.
Keeping validation here makes the accepted input contract easier to
reason about and avoids duplicating checks across compilation modules.

Role in the package
-------------------
This is an internal validation module for graph compilation. It defines
what inputs are accepted by compile_graph() and raises clear errors for
unsupported or ambiguous usage. It should not contain graph parsing,
backend compilation, or PyTorch model construction logic.
"""

from typing import Any

import pandas as pd

from ..utils.constants import INTERNAL_NODE_PREFIX
from ..utils.errors import Edge2TorchError


def validate_compile_graph_inputs(
    edgelist: Any,
    backend: Any,
    quiet: Any,
) -> None:
    """
    Validate the public inputs of ``compile_graph()``.

    Parameters
    ----------
    edgelist
        Edge table with required columns ``source`` and ``target``.
    backend
        The backend to compile to.
    quiet
        Whether informational notes should be suppressed.

    Raises
    ------
    Edge2TorchError
        If any input is invalid.
    """
    # Edgelist validation
    if not isinstance(edgelist, pd.DataFrame):
        raise Edge2TorchError("'edgelist' must be a pandas DataFrame.")

    required_columns = {"source", "target"}
    missing_columns = required_columns.difference(edgelist.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise Edge2TorchError(
            "'edgelist' must contain the columns 'source' and "
            f"'target'. Missing: {missing_str}."
        )

    source_count = int((edgelist.columns == "source").sum())
    target_count = int((edgelist.columns == "target").sum())

    if source_count != 1 or target_count != 1:
        raise Edge2TorchError(
            "'edgelist' columns 'source' and 'target' must each appear "
            "exactly once."
        )

    edge_columns = edgelist.loc[:, ["source", "target"]]

    if edge_columns.isnull().any().any():
        raise Edge2TorchError(
            "'edgelist' columns 'source' and 'target' must not "
            "contain missing values."
        )

    normalized_sources = edge_columns["source"].astype(str).str.strip()
    normalized_targets = edge_columns["target"].astype(str).str.strip()

    source_empty = normalized_sources.eq("")
    target_empty = normalized_targets.eq("")

    if source_empty.any() or target_empty.any():
        raise Edge2TorchError(
            "'edgelist' columns 'source' and 'target' must not contain "
            "empty node names."
        )

    reserved_sources = normalized_sources.str.startswith(INTERNAL_NODE_PREFIX)
    reserved_targets = normalized_targets.str.startswith(INTERNAL_NODE_PREFIX)

    if reserved_sources.any() or reserved_targets.any():
        raise Edge2TorchError(
            "Node names starting with "
            f"'{INTERNAL_NODE_PREFIX}' are reserved for internal "
            "edge2torch nodes."
        )

    # Backend validation
    if not isinstance(backend, str):
        raise Edge2TorchError("'backend' must be a string.")

    supported_backends = {"feedforward", "recurrent", "graphnn"}

    if backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise Edge2TorchError(
            f"Unsupported backend '{backend}'. Expected one of: {supported}."
        )

    # Verbosity validation
    if not isinstance(quiet, bool):
        raise Edge2TorchError(
            "'quiet' must be a boolean value (True or False)."
        )
