from __future__ import annotations

from typing import Any

import anndata as ad
import pandas as pd
import torch

from .errors import KPNNError
from .interpret.captum_adapter import run_captum_interpretation
from .interpret.prepare_input import prepare_interpretation_input
from .interpret.validate import validate_interpretation_input


def interpret_model(
    model: Any,
    artifact: Any,
    data: pd.DataFrame | ad.AnnData | torch.Tensor,
    target: str = "nodes",
    method: str = "layer_conductance",
    quiet: bool = False,
):
    """
    Interpret a compiled KPNN model with a Captum-based method.

    Parameters
    ----------
    model : Any
        Trained PyTorch model returned by `compile_graph`.
    artifact : Any
        Compilation artifact returned by `compile_graph`.
    data : pd.DataFrame | anndata.AnnData | torch.Tensor
        Input data used for attribution.
    target : str, default="nodes"
        Interpretation target. One of: "nodes", "features".
    method : str, default="layer_conductance"
        Attribution method. Must be compatible with `target`.
    quiet : bool, default=False
        If False, emit informational notes. If True, suppress notes and only
        surface warnings and errors.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        If `target="features"`, returns one DataFrame with rows as examples
        and columns as feature names.

        If `target="nodes"`, returns a dictionary mapping layer names to
        DataFrames. Each DataFrame has rows as examples and columns as node
        names for that layer.

    Raises
    ------
    KPNNError
        If interpretation input validation fails.
    """
    validate_interpretation_input(
        model=model,
        artifact=artifact,
        data=data,
        target=target,
        method=method,
    )

    prepared_input = prepare_interpretation_input(
        data=data,
        artifact=artifact,
    )

    result = run_captum_interpretation(
        model=model,
        artifact=artifact,
        inputs=prepared_input.inputs,
        sample_names=prepared_input.sample_names,
        feature_names=prepared_input.feature_names,
        target=target,
        method=method,
        quiet=quiet,
    )

    if not quiet:
        print(
            f"[kpnn] Finished interpretation with method '{method}' "
            f"for target '{target}'."
        )

    return result
