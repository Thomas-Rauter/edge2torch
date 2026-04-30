"""
Captum method registry for interpretation dispatch and validation.

Why this file exists
--------------------
This file centralizes the mapping between kpnn method names and Captum
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
    KernelShap,
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

NODE_METHODS = {
    "layer_conductance",
    "layer_integrated_gradients",
}

SUPPORTED_METHODS = FEATURE_METHODS | NODE_METHODS
