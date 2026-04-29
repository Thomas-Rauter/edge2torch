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

import pandas as pd

from ..utils.errors import KPNNError


def validate_compile_graph_inputs(
    edgelist,
    backend,
    quiet,
):
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
    KPNNError
        If any input is invalid.
    """
    if not isinstance(edgelist, pd.DataFrame):
        raise KPNNError("'edgelist' must be a pandas DataFrame.")

    required_columns = {"source", "target"}
    missing_columns = required_columns.difference(edgelist.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise KPNNError(
            "'edgelist' must contain the columns 'source' and "
            f"'target'. Missing: {missing_str}."
        )

    supported_backends = {"feedforward", "recurrent", "graphnn"}

    if backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise KPNNError(
            f"Unsupported backend '{backend}'. Expected one of: {supported}."
        )

    if not isinstance(quiet, bool):
        raise KPNNError("'quiet' must be a boolean value (True or False).")

    edge_columns = edgelist.loc[:, ["source", "target"]]

    if edge_columns.isnull().any().any():
        raise KPNNError(
            "'edgelist' columns 'source' and 'target' must not "
            "contain missing values."
        )

    source_empty = edge_columns["source"].astype(str).str.strip().eq("")
    target_empty = edge_columns["target"].astype(str).str.strip().eq("")

    if source_empty.any() or target_empty.any():
        raise KPNNError(
            "'edgelist' columns 'source' and 'target' must not contain "
            "empty node names."
        )
