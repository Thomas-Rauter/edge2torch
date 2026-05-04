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
    "integrated_gradients": IntegratedGradients,
    "deeplift": DeepLift,
    "deeplift_shap": DeepLiftShap,
    "gradient_shap": GradientShap,
}

FEATURE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS = {
    "saliency": Saliency,
    "input_x_gradient": InputXGradient,
    "guided_backprop": GuidedBackprop,
    "deconvolution": Deconvolution,
    "feature_ablation": FeatureAblation,
    "occlusion": Occlusion,
    "feature_permutation": FeaturePermutation,
    "shapley_value_sampling": ShapleyValueSampling,
    "shapley_values": ShapleyValues,
    "lime": Lime,
    "kernel_shap": KernelShap,
    "lrp": LRP,
}

FEATURE_METHODS = set(FEATURE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS) | set(
    FEATURE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS
)

FEEDFORWARD_NODE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS = {
    "layer_conductance": LayerConductance,
    "internal_influence": InternalInfluence,
    "layer_gradient_x_activation": LayerGradientXActivation,
    "layer_deeplift": LayerDeepLift,
    "layer_deeplift_shap": LayerDeepLiftShap,
    "layer_gradient_shap": LayerGradientShap,
    "layer_integrated_gradients": LayerIntegratedGradients,
}

FEEDFORWARD_NODE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS = {
    "layer_activation": LayerActivation,
    "layer_feature_ablation": LayerFeatureAblation,
    "layer_feature_permutation": LayerFeaturePermutation,
    "layer_lrp": LayerLRP,
}

FEEDFORWARD_NODE_METHODS = set(
    FEEDFORWARD_NODE_INTERPRETERS_WITH_CONSTRUCTOR_KWARGS
) | set(FEEDFORWARD_NODE_INTERPRETERS_WITHOUT_CONSTRUCTOR_KWARGS)

SUPPORTED_METHODS = FEATURE_METHODS | FEEDFORWARD_NODE_METHODS
