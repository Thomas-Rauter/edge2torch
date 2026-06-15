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
from .site_node_attribution import run_site_node_attribution

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
    *,
    nodes: str = "non_input",
    level: str = "sites",
    site_aggregation: str = "max_abs",
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
    nodes
        Node filter for ``target="nodes"``.
    level
        Node interpretation detail level.
    site_aggregation
        Aggregation rule for summary node interpretation in recurrent and
        graphnn backends.

    Notes
    -----
    The model is temporarily switched to evaluation mode during interpretation.
    Its original training/evaluation state is restored before returning.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Feature or node attributions mapped back to named entities.

    Raises
    ------
    Edge2TorchError
        If interpretation fails or outputs have unexpected shape.
    """
    was_training = model.training
    model.eval()

    try:
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
            return run_site_node_attribution(
                model=model,
                artifact=artifact,
                inputs=inputs,
                sample_names=sample_names,
                method=method,
                constructor_kwargs=constructor_kwargs,
                attribute_kwargs=attribute_kwargs,
                nodes=nodes,  # type: ignore[arg-type]
                level=level,  # type: ignore[arg-type]
                site_aggregation=site_aggregation,  # type: ignore[arg-type]
            )

        raise Edge2TorchError(f"Unsupported interpretation target '{target}'.")

    finally:
        model.train(was_training)
