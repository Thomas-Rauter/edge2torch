"""
Feature-level Captum attribution methods.

Why this file exists
--------------------
This file contains Captum method implementations that attribute model outputs
to input features. These methods are backend-agnostic as long as the compiled
model behaves as a differentiable PyTorch function from input tensor to output
tensor.

Role in the package
-------------------
This is an internal interpretation module. It should implement feature-level
Captum attribution and map outputs back to feature names. It should not handle
public API validation, graph compilation, input preparation, or plotting.
"""

from typing import Any, cast

import pandas as pd
import torch
from torch import nn

from ..compile.artifact import CompileArtifact
from ..utils.constants import COMPILE_BACKENDS
from ..utils.errors import Edge2TorchError
from .captum_classes import get_captum_class
from .method_registry import (
    FEATURE_METHODS_WITH_CONSTRUCTOR_KWARGS,
    FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS,
)

# Level 2 functions (called by level 1 functions) ------------------------------


def run_feature_attribution(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    feature_names: list[str],
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Run feature-level attribution and return one DataFrame.

    This path is backend-agnostic as long as the compiled model supports
    standard differentiable PyTorch input-output behavior.
    """
    if artifact.backend not in COMPILE_BACKENDS:
        raise Edge2TorchError(
            f"Unsupported backend '{artifact.backend}' for "
            "feature interpretation."
        )

    interpreter = _build_feature_interpreter(
        method=method,
        model=model,
        constructor_kwargs=constructor_kwargs,
    )

    interpreter = cast(Any, interpreter)
    attributions = interpreter.attribute(
        inputs,
        **attribute_kwargs,
    )

    attributions = _normalize_feature_attributions(attributions)

    _validate_feature_attributions(
        attributions=attributions,
        sample_names=sample_names,
        feature_names=feature_names,
    )

    return pd.DataFrame(
        attributions.detach().cpu().numpy(),
        index=sample_names,
        columns=feature_names,
    )


# Level 3 functions (called by level 2 functions) ------------------------------


def _build_feature_interpreter(
    method: str,
    model: nn.Module,
    constructor_kwargs: dict[str, Any],
):
    """
    Build a Captum interpreter for feature-level attribution.
    """
    if method in FEATURE_METHODS_WITH_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(
            model,
            **constructor_kwargs,
        )

    if method in FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(model)

    raise Edge2TorchError(
        f"Method '{method}' is not supported for target='features'."
    )


def _normalize_feature_attributions(
    attributions: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize supported Captum feature-attribution output shapes.

    Most feature-attribution methods return shape
    ``(n_examples, n_features)``. Some methods may return
    ``(n_examples, 1, n_features)`` for single-output models; this singleton
    output dimension is removed so the result can be represented as one
    feature-attribution DataFrame.
    """
    if attributions.ndim == 3 and attributions.shape[1] == 1:
        return attributions.squeeze(1)

    return attributions


def _validate_feature_attributions(
    attributions: torch.Tensor,
    sample_names: list[str],
    feature_names: list[str],
) -> None:
    """
    Validate feature-attribution output shape.
    """
    if attributions.ndim != 2:
        raise Edge2TorchError(
            "Feature attributions must have shape (n_examples, n_features)."
        )

    n_examples, n_features = attributions.shape

    if n_examples != len(sample_names):
        raise Edge2TorchError(
            "Feature attribution row count does not match sample count."
        )

    if n_features != len(feature_names):
        raise Edge2TorchError(
            "Feature attribution column count does not match feature count."
        )
