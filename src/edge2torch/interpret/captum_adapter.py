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

from typing import Any, Union

import pandas as pd
import torch
from torch import nn

from ..compile.artifact import CompileArtifact
from ..utils.errors import Edge2TorchError
from .feature_attribution import run_feature_attribution
from .feedforward_node_attribution import run_feedforward_node_attribution

# Level 1 functions (called by API functions) ----------------------------------


def run_captum_interpretation(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    feature_names: list[str],
    target: str,
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
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
    constructor_kwargs
        Keyword arguments passed to the selected Captum attribution class
        constructor.
    attribute_kwargs
        Keyword arguments passed to the selected Captum ``attribute()`` call.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Feature or node attributions mapped back to named entities.

    Raises
    ------
    Edge2TorchError
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
            constructor_kwargs=constructor_kwargs,
            attribute_kwargs=attribute_kwargs,
        )

    if target == "nodes":
        return _run_node_interpretation(
            model=model,
            artifact=artifact,
            inputs=inputs,
            sample_names=sample_names,
            method=method,
            constructor_kwargs=constructor_kwargs,
            attribute_kwargs=attribute_kwargs,
        )

    raise Edge2TorchError(f"Unsupported interpretation target '{target}'.")


# Level 2 functions (called by level 1 functions) ------------------------------


def _run_node_interpretation(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
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
            constructor_kwargs=constructor_kwargs,
            attribute_kwargs=attribute_kwargs,
        )

    if backend == "recurrent":
        raise Edge2TorchError(
            "Node interpretation is not yet implemented for the "
            "'recurrent' backend."
        )

    if backend == "graphnn":
        raise Edge2TorchError(
            "Node interpretation is not yet implemented for the "
            "'graphnn' backend."
        )

    raise Edge2TorchError(
        f"Unsupported backend '{backend}' for target='nodes'."
    )
