"""
Captum method registry for interpretation dispatch and validation.

Why this file exists
--------------------
This file centralizes the mapping between edge2torch method names and Captum
attribution classes. Keeping method names in one place prevents validation and
execution code from drifting apart as supported Captum methods are added.

Role in the package
-------------------
This is an internal interpretation registry module. It should define supported
method names and Captum class mappings. It should not perform validation,
execute attribution methods, prepare input data, or map attribution results.
"""

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

FEATURE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS = {
    "IntegratedGradients": IntegratedGradients,
    "DeepLift": DeepLift,
    "DeepLiftShap": DeepLiftShap,
    "GradientShap": GradientShap,
}

FEATURE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS = {
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
}

FEATURE_METHODS = set(FEATURE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS) | set(
    FEATURE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS
)

FEEDFORWARD_NODE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS = {
    "LayerConductance": LayerConductance,
    "InternalInfluence": InternalInfluence,
    "LayerGradientXActivation": LayerGradientXActivation,
    "LayerDeepLift": LayerDeepLift,
    "LayerDeepLiftShap": LayerDeepLiftShap,
    "LayerGradientShap": LayerGradientShap,
    "LayerIntegratedGradients": LayerIntegratedGradients,
}

FEEDFORWARD_NODE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS = {
    "LayerActivation": LayerActivation,
    "LayerFeatureAblation": LayerFeatureAblation,
    "LayerFeaturePermutation": LayerFeaturePermutation,
    "LayerLRP": LayerLRP,
}

FEEDFORWARD_NODE_METHODS = set(
    FEEDFORWARD_NODE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS
) | set(FEEDFORWARD_NODE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS)

SUPPORTED_METHODS = FEATURE_METHODS | FEEDFORWARD_NODE_METHODS
