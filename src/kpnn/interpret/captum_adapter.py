"""
Captum-based interpretation dispatch and result mapping.

Why this file exists
--------------------
This file isolates the package's integration with Captum so that
attribution-method handling, backend-aware dispatch, and result mapping
are kept separate from the public interpret_model() API. The separation
makes it easier to extend supported interpretation paths without mixing
Captum-specific logic into validation or API orchestration.

Role in the package
-------------------
This is an internal interpretation-execution module. It chooses the
appropriate Captum-based interpretation path for a validated request,
runs the underlying attribution method, and maps outputs back to
artifact-defined names. It should contain backend- and method-specific
interpretation logic, not public API validation, input preparation, or
downstream plotting and analysis.
"""

import pandas as pd
from captum.attr import (
    IntegratedGradients,
    LayerConductance,
    LayerIntegratedGradients,
)

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
    quiet
        Whether informational notes should be suppressed.

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
        return _run_feature_interpretation(
            model=model,
            artifact=artifact,
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
            method=method,
        )

    if target == "nodes":
        backend = artifact.backend

        if backend == "feedforward":
            return _run_feedforward_node_interpretation(
                model=model,
                artifact=artifact,
                inputs=inputs,
                sample_names=sample_names,
                method=method,
            )

        if backend == "recurrent":
            return _run_recurrent_node_interpretation(
                model=model,
                artifact=artifact,
                inputs=inputs,
                sample_names=sample_names,
                method=method,
            )

        if backend == "graphnn":
            return _run_graphnn_node_interpretation(
                model=model,
                artifact=artifact,
                inputs=inputs,
                sample_names=sample_names,
                method=method,
            )

        raise KPNNError(
            f"Unsupported backend '{backend}' for target='nodes'."
        )

    raise KPNNError(
        f"Unsupported interpretation target '{target}'."
    )


def _run_feature_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    feature_names,
    method,
):
    """
    Run feature-level attribution and return one DataFrame.

    This path is backend-agnostic as long as the compiled model supports
    standard differentiable PyTorch input-output behavior.
    """
    if method != "integrated_gradients":
        raise KPNNError(
            f"Method '{method}' is not supported for "
            "target='features'."
        )

    if artifact.backend not in {
        "feedforward",
        "recurrent",
        "graphnn",
    }:
        raise KPNNError(
            f"Unsupported backend '{artifact.backend}' for "
            "feature interpretation."
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
            "Feature attribution row count does not match sample count."
        )

    if n_features != len(feature_names):
        raise KPNNError(
            "Feature attribution column count does not match "
            "feature count."
        )

    return pd.DataFrame(
        attributions.detach().cpu().numpy(),
        index=sample_names,
        columns=feature_names,
    )


def _run_feedforward_node_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    method,
):
    """
    Run feedforward node-level attribution and return one DataFrame per
    layer.
    """
    if method not in {
        "layer_conductance",
        "layer_integrated_gradients",
    }:
        raise KPNNError(
            f"Method '{method}' is not supported for target='nodes'."
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
                "does not match sample count."
            )

        if n_nodes != len(node_names):
            raise KPNNError(
                f"Node attribution width mismatch for '{layer_name}'. "
                f"Expected {len(node_names)}, got {n_nodes}."
            )

        visible_indices = [
            idx
            for idx, node_name in enumerate(node_names)
            if _is_visible_biological_node(node_name)
        ]
        visible_node_names = [node_names[idx] for idx in visible_indices]

        visible_attributions = attributions[:, visible_indices]

        results[layer_name] = pd.DataFrame(
            visible_attributions.detach().cpu().numpy(),
            index=sample_names,
            columns=visible_node_names,
        )

    return results


def _run_recurrent_node_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    method,
):
    """
    Placeholder for recurrent node-level interpretation.
    """
    raise KPNNError(
        "Node interpretation is not yet implemented for the "
        "'recurrent' backend."
    )


def _run_graphnn_node_interpretation(
    model,
    artifact,
    inputs,
    sample_names,
    method,
):
    """
    Placeholder for graphnn node-level interpretation.
    """
    raise KPNNError(
        "Node interpretation is not yet implemented for the "
        "'graphnn' backend."
    )


def _is_visible_biological_node(node_name: str) -> bool:
    """
    Return True if a node should be exposed in interpretation output.
    """
    return not node_name.startswith("pseudo__")


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
