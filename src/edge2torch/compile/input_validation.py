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

import numpy as np
import pandas as pd

from ..utils.constants import INTERNAL_NODE_PREFIX
from ..utils.errors import Edge2TorchError

# Level 1 functions (functions called by API functions) ------------------------


def validate_compile_graph_inputs(
    edgelist: Any,
    backend: Any,
    quiet: Any,
    bias: Any,
) -> None:
    """
    Validate the public inputs of ``compile_graph()``.

    Parameters
    ----------
    edgelist
        Edge table with required columns ``source`` and ``target``.
        Optionally, the table may include sparse edge-level metadata in
        ``initial_weight`` and/or ``constraint``.
    backend
        The backend to compile to.
    quiet
        Whether informational notes should be suppressed.
    bias
        Whether compiled masked linear layers should include bias terms.

    Raises
    ------
    Edge2TorchError
        If any input is invalid.
    """
    _validate_edgelist_dataframe(edgelist)
    _validate_required_edge_columns(edgelist)
    _validate_source_target_values(edgelist)
    _validate_optional_edge_metadata(edgelist)
    _validate_backend(backend)
    _validate_quiet(quiet)
    _validate_bias(bias)


# Level 2 functions (functions called by level 1 functions) --------------------


def _validate_edgelist_dataframe(edgelist: Any) -> None:
    """
    Validate that the edgelist is a pandas DataFrame.
    """
    if not isinstance(edgelist, pd.DataFrame):
        raise Edge2TorchError("'edgelist' must be a pandas DataFrame.")


def _validate_required_edge_columns(edgelist: pd.DataFrame) -> None:
    """
    Validate required edgelist columns and duplicate reserved column names.
    """
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

    initial_weight_count = int((edgelist.columns == "initial_weight").sum())
    constraint_count = int((edgelist.columns == "constraint").sum())

    if initial_weight_count > 1 or constraint_count > 1:
        raise Edge2TorchError(
            "'edgelist' columns 'initial_weight' and 'constraint' must each "
            "appear at most once."
        )


def _validate_source_target_values(edgelist: pd.DataFrame) -> None:
    """
    Validate source and target node values.
    """
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


def _validate_optional_edge_metadata(edgelist: pd.DataFrame) -> None:
    """
    Validate optional sparse edge-level initial weights and constraints.

    ``initial_weight`` and ``constraint`` are independent optional columns.

    If ``initial_weight`` is present, missing values are allowed and mean that
    the corresponding edge should use default initialization. Non-missing
    values must be finite numeric values.

    If ``constraint`` is present, missing values are allowed and mean that the
    corresponding edge is unconstrained. Non-missing values must be one of
    ``unconstrained``, ``positive``, ``negative``, or ``fixed``.

    If ``constraint`` is ``fixed`` for a row, that row must have an explicit
    ``initial_weight`` value because fixed edges need a defined constant value.

    If both metadata values are present in a row, positive-constrained edges
    must have positive initial weights, and negative-constrained edges must
    have negative initial weights.
    """
    has_initial_weight = "initial_weight" in edgelist.columns
    has_constraint = "constraint" in edgelist.columns

    initial_weight = None
    constraint = None

    if has_initial_weight:
        initial_weight = _validate_initial_weight_column(edgelist)

    if has_constraint:
        constraint = _validate_constraint_column(edgelist)

    if has_initial_weight or has_constraint:
        _validate_sparse_edge_metadata_consistency(
            initial_weight=initial_weight,
            constraint=constraint,
            n_edges=len(edgelist),
        )


def _validate_backend(backend: Any) -> None:
    """
    Validate backend selection.
    """
    if not isinstance(backend, str):
        raise Edge2TorchError("'backend' must be a string.")

    supported_backends = {"feedforward", "recurrent", "graphnn"}

    if backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise Edge2TorchError(
            f"Unsupported backend '{backend}'. Expected one of: {supported}."
        )


def _validate_quiet(quiet: Any) -> None:
    """
    Validate quiet flag.
    """
    if not isinstance(quiet, bool):
        raise Edge2TorchError(
            "'quiet' must be a boolean value (True or False)."
        )


def _validate_bias(bias: Any) -> None:
    """
    Validate bias flag.
    """
    if not isinstance(bias, bool):
        raise Edge2TorchError("'bias' must be a boolean value (True or False).")


# Level 3 functions (functions called by level 2 functions) --------------------


def _validate_initial_weight_column(
    edgelist: pd.DataFrame,
) -> pd.Series:
    """
    Validate and return the numeric initial_weight column.

    Missing values are allowed and mean that the corresponding edge should use
    default initialization.
    """
    initial_weight = pd.to_numeric(
        edgelist["initial_weight"],
        errors="coerce",
    )

    invalid_non_numeric = (
        initial_weight.isnull() & edgelist["initial_weight"].notnull()
    )

    if invalid_non_numeric.any():
        raise Edge2TorchError(
            "'edgelist' column 'initial_weight' must contain only numeric "
            "values or missing values."
        )

    non_missing_weight = initial_weight.dropna()

    if not np.isfinite(non_missing_weight.to_numpy(dtype=float)).all():
        raise Edge2TorchError(
            "'edgelist' column 'initial_weight' must contain only finite "
            "numeric values or missing values."
        )

    return initial_weight


def _validate_constraint_column(
    edgelist: pd.DataFrame,
) -> pd.Series:
    """
    Validate and return the normalized constraint column.

    Missing values are allowed and mean ``unconstrained``.
    """
    constraint = edgelist["constraint"].astype("string").str.strip().str.lower()

    allowed_constraints = {
        "unconstrained",
        "positive",
        "negative",
        "fixed",
    }

    non_missing_constraint = constraint.dropna()

    invalid_constraints = sorted(
        set(non_missing_constraint).difference(allowed_constraints)
    )

    if invalid_constraints:
        invalid_str = ", ".join(invalid_constraints)
        allowed_str = ", ".join(sorted(allowed_constraints))
        raise Edge2TorchError(
            "'edgelist' column 'constraint' contains unsupported "
            f"value(s): {invalid_str}. Expected one of: {allowed_str}, "
            "or a missing value."
        )

    return constraint


def _validate_sparse_edge_metadata_consistency(
    initial_weight: pd.Series | None,
    constraint: pd.Series | None,
    n_edges: int,
) -> None:
    """
    Validate row-wise consistency between optional initial weights and
    optional constraints.
    """
    if initial_weight is None:
        initial_weight = pd.Series([pd.NA] * n_edges, dtype="Float64")

    if constraint is None:
        constraint = pd.Series(["unconstrained"] * n_edges, dtype="string")
    else:
        constraint = constraint.fillna("unconstrained")

    _validate_fixed_edges_have_initial_weight(
        initial_weight=initial_weight,
        constraint=constraint,
    )

    _validate_initial_weight_matches_constraint(
        initial_weight=initial_weight,
        constraint=constraint,
    )


def _validate_fixed_edges_have_initial_weight(
    initial_weight: pd.Series,
    constraint: pd.Series,
) -> None:
    """
    Validate that fixed edges have explicit initial weights.
    """
    fixed_without_initial_weight = (
        constraint.eq("fixed") & initial_weight.isnull()
    )

    if fixed_without_initial_weight.any():
        raise Edge2TorchError(
            "Edges with constraint 'fixed' require an 'initial_weight' "
            "value in the same row because fixed edges need an explicit "
            "constant value."
        )


def _validate_initial_weight_matches_constraint(
    initial_weight: pd.Series,
    constraint: pd.Series,
) -> None:
    """
    Validate consistency between provided initial weights and constraints.
    """
    has_initial_weight = initial_weight.notnull()

    positive_with_nonpositive_weight = (
        constraint.eq("positive") & has_initial_weight & initial_weight.le(0)
    )

    if positive_with_nonpositive_weight.any():
        raise Edge2TorchError(
            "Edges with constraint 'positive' must have "
            "'initial_weight' > 0 when 'initial_weight' is provided."
        )

    negative_with_nonnegative_weight = (
        constraint.eq("negative") & has_initial_weight & initial_weight.ge(0)
    )

    if negative_with_nonnegative_weight.any():
        raise Edge2TorchError(
            "Edges with constraint 'negative' must have "
            "'initial_weight' < 0 when 'initial_weight' is provided."
        )
