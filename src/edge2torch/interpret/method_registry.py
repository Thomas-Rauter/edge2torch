"""
Captum method registry for interpretation dispatch and validation.

Why this file exists
--------------------
This file centralizes the supported Captum method names used by edge2torch.
Keeping method names in one place prevents validation and execution code from
drifting apart as supported interpretation methods are added.

Role in the package
-------------------
This is an internal interpretation registry module. It defines supported method
names and method categories. It should not import Captum classes, perform
validation, execute attribution methods, prepare input data, or map attribution
results.
"""

FEATURE_METHODS_WITH_CONSTRUCTOR_KWARGS = {
    "IntegratedGradients",
    "DeepLift",
    "DeepLiftShap",
    "GradientShap",
}

FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS = {
    "Saliency",
    "InputXGradient",
    "GuidedBackprop",
    "Deconvolution",
    "FeatureAblation",
    "Occlusion",
    "FeaturePermutation",
    "ShapleyValueSampling",
    "ShapleyValues",
    "Lime",
    "KernelShap",
    "LRP",
}

FEATURE_METHODS = (
    FEATURE_METHODS_WITH_CONSTRUCTOR_KWARGS
    | FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS
)

NODE_METHODS_WITH_CONSTRUCTOR_KWARGS = {
    "LayerConductance",
    "InternalInfluence",
    "LayerGradientXActivation",
    "LayerDeepLift",
    "LayerDeepLiftShap",
    "LayerGradientShap",
    "LayerIntegratedGradients",
}

NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS = {
    "LayerActivation",
    "LayerFeatureAblation",
    "LayerFeaturePermutation",
    "LayerLRP",
}

NODE_METHODS = (
    NODE_METHODS_WITH_CONSTRUCTOR_KWARGS
    | NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS
)

FEEDFORWARD_NODE_METHODS_WITH_CONSTRUCTOR_KWARGS = (
    NODE_METHODS_WITH_CONSTRUCTOR_KWARGS
)

FEEDFORWARD_NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS = (
    NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS
)

FEEDFORWARD_NODE_METHODS = NODE_METHODS

SUPPORTED_METHODS = FEATURE_METHODS | NODE_METHODS
