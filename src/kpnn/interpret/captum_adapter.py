import pandas as pd
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import LayerIntegratedGradients

from ..utils.errors import KPNNError


def run_captum_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    feature_names,
    target,
    method,
    quiet,
):
    """
    Run a Captum interpretation and map results back to biological names.

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
    quiet
        Whether informational notes should be suppressed.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Feature or node attributions mapped back to biological names.

    Raises
    ------
    KPNNError
        If interpretation fails or outputs have unexpected shape.
    """
    if target == "features":
        return _run_feature_interpretation(
            model=model,
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

    raise KPNNError(
        f"Unsupported interpretation target '{target}'."
    )


def _run_feature_interpretation(
    model,
    inputs,
    sample_names,
    feature_names,
    method,
):
    """
    Run feature-level attribution and return one DataFrame.
    """
    if method != "integrated_gradients":
        raise KPNNError(
            f"Method '{method}' is not supported for "
            "target='features'."
        )

    interpreter = IntegratedGradients(model)
    attributions = interpreter.attribute(inputs)

    if attributions.ndim != 2:
        raise KPNNError(
            "Feature attributions must have shape "
            "(n_examples, n_features)."
        )

    n_examples, n_features = attributions.shape

    if n_examples != len(sample_names):
        raise KPNNError(
            "Feature attribution row count does not match "
            "the number of samples."
        )

    if n_features != len(feature_names):
        raise KPNNError(
            "Feature attribution column count does not match "
            "the number of features."
        )

    return pd.DataFrame(
        attributions.detach().cpu().numpy(),
        index=sample_names,
        columns=feature_names,
    )


def _run_node_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    method,
):
    """
    Run node-level attribution and return one DataFrame per layer.
    """
    if method not in {
        "layer_conductance",
        "layer_integrated_gradients",
    }:
        raise KPNNError(
            f"Method '{method}' is not supported for "
            "target='nodes'."
        )

    results = {}

    layer_names = sorted(
        artifact.node_names_by_layer.keys(),
        key=_layer_sort_key,
    )

    for layer_name in layer_names:
        if layer_name == "layer_0":
            continue

        node_names = artifact.node_names_by_layer[layer_name]
        layer_block = model.get_layer_block(layer_name)

        if method == "layer_conductance":
            interpreter = LayerConductance(model, layer_block)
        else:
            interpreter = LayerIntegratedGradients(model, layer_block)

        attributions = interpreter.attribute(inputs)

        if attributions.ndim != 2:
            raise KPNNError(
                f"Node attributions for '{layer_name}' must have "
                "shape (n_examples, n_nodes)."
            )

        n_examples, n_nodes = attributions.shape

        if n_examples != len(sample_names):
            raise KPNNError(
                f"Node attribution row count for '{layer_name}' "
                "does not match the number of samples."
            )

        if n_nodes != len(node_names):
            raise KPNNError(
                f"Node attribution width mismatch for '{layer_name}'. "
                f"Expected {len(node_names)}, got {n_nodes}."
            )

        results[layer_name] = pd.DataFrame(
            attributions.detach().cpu().numpy(),
            index=sample_names,
            columns=node_names,
        )

    return results


def _layer_sort_key(layer_name):
    """
    Sort layer names like 'layer_0', 'layer_1', ...
    """
    try:
        return int(layer_name.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise KPNNError(
            f"Invalid layer name '{layer_name}' in artifact."
        ) from exc
