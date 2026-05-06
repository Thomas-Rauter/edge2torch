"""
Lazy Captum class resolution for optional interpretation support.

Why this file exists
--------------------
Captum is an optional dependency of edge2torch. This module imports Captum
classes only when interpretation is actually executed.

Role in the package
-------------------
This is an internal dependency bridge. It maps validated Captum method names to
Captum attribution classes. It should not validate public inputs, execute
attribution, prepare input data, or map attribution results.
"""

from typing import Any

from ..utils.errors import Edge2TorchError


def get_captum_class(method: str) -> type[Any]:
    """
    Return the Captum attribution class for a supported method name.
    """
    try:
        from captum.attr import (
            LRP,
            Deconvolution,
            DeepLift,
            DeepLiftShap,
            FeatureAblation,
            FeaturePermutation,
            GradientShap,
            GuidedBackprop,
            InputXGradient,
            IntegratedGradients,
            InternalInfluence,
            KernelShap,
            LayerActivation,
            LayerConductance,
            LayerDeepLift,
            LayerDeepLiftShap,
            LayerFeatureAblation,
            LayerFeaturePermutation,
            LayerGradientShap,
            LayerGradientXActivation,
            LayerIntegratedGradients,
            LayerLRP,
            Lime,
            Occlusion,
            Saliency,
            ShapleyValues,
            ShapleyValueSampling,
        )
    except ImportError as exc:
        raise Edge2TorchError(
            "interpret_model() requires Captum, which is an optional "
            "dependency. Install interpretation support with "
            "'pip install \"edge2torch[interpret]\"'."
        ) from exc

    classes: dict[str, type[Any]] = {
        "IntegratedGradients": IntegratedGradients,
        "DeepLift": DeepLift,
        "DeepLiftShap": DeepLiftShap,
        "GradientShap": GradientShap,
        "Saliency": Saliency,
        "InputXGradient": InputXGradient,
        "GuidedBackprop": GuidedBackprop,
        "Deconvolution": Deconvolution,
        "FeatureAblation": FeatureAblation,
        "Occlusion": Occlusion,
        "FeaturePermutation": FeaturePermutation,
        "ShapleyValueSampling": ShapleyValueSampling,
        "ShapleyValues": ShapleyValues,
        "Lime": Lime,
        "KernelShap": KernelShap,
        "LRP": LRP,
        "LayerActivation": LayerActivation,
        "LayerConductance": LayerConductance,
        "InternalInfluence": InternalInfluence,
        "LayerGradientXActivation": LayerGradientXActivation,
        "LayerDeepLift": LayerDeepLift,
        "LayerDeepLiftShap": LayerDeepLiftShap,
        "LayerGradientShap": LayerGradientShap,
        "LayerIntegratedGradients": LayerIntegratedGradients,
        "LayerFeatureAblation": LayerFeatureAblation,
        "LayerFeaturePermutation": LayerFeaturePermutation,
        "LayerLRP": LayerLRP,
    }

    return classes[method]
