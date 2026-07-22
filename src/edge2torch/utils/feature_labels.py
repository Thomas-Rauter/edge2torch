"""
Helpers that normalize feature labels to string node IDs.

Why this file exists
--------------------
Graph compilation stores node and feature names as strings. Named data
containers may still use non-string labels (for example integer column
names). These helpers keep that boundary conversion in one place.

Role in the package
-------------------
Internal utilities for align and interpret input handling. Not part of
the public API.
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .errors import Edge2TorchError


def as_str_labels(labels: Iterable[Any]) -> list[str]:
    """
    Coerce labels to stripped strings.

    Parameters
    ----------
    labels
        Feature or variable labels from a data container.

    Returns
    -------
    list[str]
        Labels in the package's canonical string form.
    """
    return [str(label).strip() for label in labels]


def unique_str_labels(
    labels: Iterable[Any],
    *,
    duplicate_message: str,
) -> list[str]:
    """
    Coerce labels to strings and reject duplicates after coercion.

    Parameters
    ----------
    labels
        Feature or variable labels from a data container.
    duplicate_message
        Error text when labels are not unique after ``str`` conversion.

    Returns
    -------
    list[str]
        Unique string labels in input order.

    Raises
    ------
    Edge2TorchError
        If string coercion creates duplicate labels.
    """
    str_labels = as_str_labels(labels)

    if len(set(str_labels)) != len(str_labels):
        raise Edge2TorchError(duplicate_message)

    return str_labels


def dataframe_with_str_feature_columns(
    data: pd.DataFrame,
    *,
    duplicate_message: str,
) -> pd.DataFrame:
    """
    Return a DataFrame whose columns are canonical string feature names.

    Parameters
    ----------
    data
        Input feature table.
    duplicate_message
        Error text for duplicate names before or after ``str`` conversion.

    Returns
    -------
    pd.DataFrame
        Shallow copy with string column labels.

    Raises
    ------
    Edge2TorchError
        If column names are not unique as-is or after ``str`` conversion.
    """
    if data.columns.duplicated().any():
        raise Edge2TorchError(duplicate_message)

    str_columns = unique_str_labels(
        data.columns,
        duplicate_message=duplicate_message,
    )

    renamed = data.copy(deep=False)
    renamed.columns = str_columns
    return renamed
