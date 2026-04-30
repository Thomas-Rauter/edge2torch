"""
Captum-based interpretation dispatch.

Why this file exists
--------------------
This file routes validated interpretation requests to backend- and
target-specific Captum attribution implementations. Keeping dispatch here
separates public interpretation orchestration from method-specific Captum
execution and result mapping.

Role in the package
-------------------
This is an internal interpretation-dispatch module. It should choose the
appropriate implementation for a validated interpretation request, not contain
method-specific attribution logic, graph compilation, input preparation, or
plotting logic.
"""

from typing import Union

import pandas as pd
import torch
from torch import nn

from ..compile.artifact import KPNNArtifact
from ..utils.errors import KPNNError
from .feature_attribution import run_feature_attribution
from .feedforward_node_attribution import run_feedforward_node_attribution


def run_captum_interpretation(
    model: nn.Module,
    artifact: KPNNArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    feature_names: list[str],
    target: str,
    method: str,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run a Captum interpretation and map results back to named entities.

    Parameters
    ----------
    model
        Trained PyTorch model to interpret.
    artifact
        Compilation artifact returned by ``compile_graph()``.
    inputs
        Standardized input tensor for attribution.
    sample_names
        Names of input examples.
    feature_names
        Names of input features.
    target
        Interpretation target. One of: ``"features"``, ``"nodes"``.
    method
        Captum attribution method compatible with ``target``.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Feature or node attributions mapped back to named entities.

    Raises
    ------
    KPNNError
        If interpretation fails or outputs have unexpected shape.
    """
    if target == "features":
        return run_feature_attribution(
            model=model,
            artifact=artifact,
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
            method=method,
        )

    if target == "nodes":
        return _run_node_interpretation(
            model=model,
            artifact=artifact,
            inputs=inputs,
            sample_names=sample_names,
            method=method,
        )

    raise KPNNError(f"Unsupported interpretation target '{target}'.")


def _run_node_interpretation(
    model: nn.Module,
    artifact: KPNNArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    method: str,
) -> dict[str, pd.DataFrame]:
    """
    Dispatch node-level interpretation by backend.
    """
    backend = artifact.backend

    if backend == "feedforward":
        return run_feedforward_node_attribution(
            model=model,
            artifact=artifact,
            inputs=inputs,
            sample_names=sample_names,
            method=method,
        )

    if backend == "recurrent":
        raise KPNNError(
            "Node interpretation is not yet implemented for the "
            "'recurrent' backend."
        )

    if backend == "graphnn":
        raise KPNNError(
            "Node interpretation is not yet implemented for the "
            "'graphnn' backend."
        )

    raise KPNNError(f"Unsupported backend '{backend}' for target='nodes'.")
