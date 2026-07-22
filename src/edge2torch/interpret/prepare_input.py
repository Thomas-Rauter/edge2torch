"""
Standardization of interpretation inputs for downstream attribution.

Why this file exists
--------------------
This file isolates the step that converts accepted interpretation input
containers into one standardized tensor-based representation. The
separation keeps container-specific preparation logic out of both the
public API wrapper and the Captum execution layer, making the
interpretation pipeline easier to reason about and extend.

Role in the package
-------------------
This is an internal interpretation-input module. It defines the
standardized prepared-input object and the logic that converts validated
DataFrame, Tensor, or optional AnnData inputs into that form. It should
contain input preparation logic, not public API validation, Captum
execution, or downstream result analysis.
"""

from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch

try:
    import anndata as ad
except ImportError:
    ad = None

from ..compile.artifact import CompileArtifact
from ..utils.errors import Edge2TorchError
from ..utils.feature_labels import (
    dataframe_with_str_feature_columns,
    unique_str_labels,
)

_DF_DUPLICATE_COLUMNS_MSG = (
    "'data' DataFrame must not contain duplicate column names "
    "(including after converting labels to strings)."
)
_ANNDATA_DUPLICATE_VARS_MSG = (
    "'data.var_names' must not contain duplicates "
    "(including after converting labels to strings)."
)


@dataclass
class PreparedInterpretationInput:
    """
    Standardized interpretation input passed to the Captum adapter.
    """

    inputs: torch.Tensor
    sample_names: list[str]
    feature_names: list[str]


def prepare_interpretation_input(
    data,
    artifact: CompileArtifact,
) -> PreparedInterpretationInput:
    """
    Convert interpretation input data into a standardized tensor form.

    Parameters
    ----------
    data
        Input data for attribution.
    artifact
        Compilation artifact returned by ``compile_graph()``.

    Returns
    -------
    PreparedInterpretationInput
        Standardized tensor input plus sample and feature names.

    Raises
    ------
    Edge2TorchError
        If input preparation fails.
    """
    feature_names = list(artifact.feature_names)

    if isinstance(data, pd.DataFrame):
        named = dataframe_with_str_feature_columns(
            data,
            duplicate_message=_DF_DUPLICATE_COLUMNS_MSG,
        )
        ordered = cast(pd.DataFrame, named[feature_names])
        inputs = torch.tensor(
            ordered.to_numpy(copy=True),
            dtype=torch.float32,
        )
        sample_names = [str(idx) for idx in ordered.index]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    if ad is not None and isinstance(data, ad.AnnData):
        var_names = unique_str_labels(
            data.var_names,
            duplicate_message=_ANNDATA_DUPLICATE_VARS_MSG,
        )

        order = [var_names.index(name) for name in feature_names]
        matrix = data.X[:, order]

        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        inputs = torch.tensor(
            matrix.copy(),
            dtype=torch.float32,
        )
        sample_names = [str(name) for name in data.obs_names]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    if isinstance(data, torch.Tensor):
        inputs = data.to(dtype=torch.float32)
        sample_names = [str(i) for i in range(inputs.shape[0])]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    raise Edge2TorchError("Unsupported interpretation input type.")
