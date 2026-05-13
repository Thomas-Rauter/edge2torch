"""
Feedforward node-level Captum attribution methods.

Why this file exists
--------------------
This file contains Captum method implementations that attribute model outputs
to feedforward layer nodes. Feedforward node attribution is implemented via
Captum layer-level methods and then mapped back to named graph nodes.

Role in the package
-------------------
This is an internal interpretation module. It should implement feedforward
node-level attribution and map results back to non-internal node names. It
should not handle public API validation, graph compilation, input preparation,
or plotting.
"""

from collections.abc import Callable
from typing import Any, cast

import pandas as pd
import torch
from torch import nn

from ..compile.artifact import CompileArtifact
from ..utils.constants import INTERNAL_NODE_PREFIX
from ..utils.errors import Edge2TorchError
from .captum_classes import get_captum_class
from .method_registry import (
    FEEDFORWARD_NODE_METHODS_WITH_CONSTRUCTOR_KWARGS,
    FEEDFORWARD_NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS,
)

# Level 3 functions (called by level 2 functions) ------------------------------


def run_feedforward_node_attribution(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Run feedforward node-level attribution and return one DataFrame per layer.
    """
    results: dict[str, pd.DataFrame] = {}

    layer_names = sorted(
        artifact.node_names_by_layer.keys(),
        key=_layer_sort_key,
    )

    for layer_name in layer_names:
        if layer_name == "layer_0":
            continue

        node_names = artifact.node_names_by_layer[layer_name]

        layer_block_provider = _find_feedforward_layer_block_provider(model)

        get_layer_block = cast(
            Callable[[str], nn.Module],
            getattr(
                layer_block_provider,
                "_edge2torch_get_feedforward_layer_block",
            ),
        )
        layer_block = get_layer_block(layer_name)

        interpreter = _build_feedforward_layer_interpreter(
            method=method,
            model=model,
            layer_block=layer_block,
            constructor_kwargs=constructor_kwargs,
        )

        interpreter = cast(Any, interpreter)
        attributions = interpreter.attribute(
            inputs,
            **attribute_kwargs,
        )

        _validate_node_attributions(
            attributions=attributions,
            layer_name=layer_name,
            sample_names=sample_names,
            node_names=node_names,
        )

        visible_indices = [
            idx
            for idx, node_name in enumerate(node_names)
            if _is_visible_domain_node(node_name)
        ]
        visible_node_names = [node_names[idx] for idx in visible_indices]

        visible_attributions = attributions[:, visible_indices]

        results[layer_name] = pd.DataFrame(
            visible_attributions.detach().cpu().numpy(),
            index=sample_names,
            columns=visible_node_names,
        )

    return results


# Level 4 functions (called by level 3 functions) ------------------------------


def _find_feedforward_layer_block_provider(model: nn.Module) -> nn.Module:
    """
    Find a module that exposes edge2torch feedforward layer-block access.

    Raw feedforward models expose this method directly. Models returned by
    ``customize_model()`` and many manually wrapped PyTorch models expose it
    through a registered submodule.
    """
    method_name = "_edge2torch_get_feedforward_layer_block"

    if hasattr(model, method_name):
        return model

    for module in model.modules():
        if module is model:
            continue

        if hasattr(module, method_name):
            return module

    raise Edge2TorchError(
        "Node-level interpretation requires access to the compiled "
        "feedforward model's internal layer blocks. This is supported for raw "
        "feedforward models returned by compile_graph(), models returned by "
        "customize_model(), and wrappers that keep the compiled model as a "
        "registered PyTorch submodule."
    )


def _build_feedforward_layer_interpreter(
    method: str,
    model: nn.Module,
    layer_block: nn.Module,
    constructor_kwargs: dict[str, Any],
):
    """
    Build a Captum interpreter for feedforward layer-level attribution.
    """
    if method in FEEDFORWARD_NODE_METHODS_WITH_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(
            model,
            layer_block,
            **constructor_kwargs,
        )

    if method in FEEDFORWARD_NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(
            model,
            layer_block,
        )

    raise Edge2TorchError(
        f"Method '{method}' is not supported for target='nodes'."
    )


def _validate_node_attributions(
    attributions: torch.Tensor,
    layer_name: str,
    sample_names: list[str],
    node_names: list[str],
) -> None:
    """
    Validate node-attribution output shape for one layer.
    """
    if attributions.ndim != 2:
        raise Edge2TorchError(
            f"Node attributions for '{layer_name}' must have "
            "shape (n_examples, n_nodes)."
        )

    n_examples, n_nodes = attributions.shape

    if n_examples != len(sample_names):
        raise Edge2TorchError(
            f"Node attribution row count for '{layer_name}' "
            "does not match sample count."
        )

    if n_nodes != len(node_names):
        raise Edge2TorchError(
            f"Node attribution width mismatch for '{layer_name}'. "
            f"Expected {len(node_names)}, got {n_nodes}."
        )


def _is_visible_domain_node(node_name: str) -> bool:
    """
    Return True if a node should be exposed in interpretation output.
    """
    return not node_name.startswith(INTERNAL_NODE_PREFIX)


def _layer_sort_key(layer_name: str) -> int:
    """
    Sort layer names like 'layer_0', 'layer_1', ...
    """
    try:
        return int(layer_name.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise Edge2TorchError(
            f"Invalid layer name '{layer_name}' in artifact."
        ) from exc
