"""
Interpretation metadata helpers for compiled KPNN artifacts.

Why this file exists
--------------------
This file centralizes construction of interpretation-related metadata that
is stored on ``CompileArtifact``. Keeping this logic separate from backend
compilers makes the site and node-role contracts easier to test and evolve.

Role in the package
-------------------
This is an internal compilation helper module. It should build interpretation
site maps and node role lists from execution plans, not perform validation,
model construction, or public API orchestration.
"""

from ..utils.constants import PSEUDO_NODE_PREFIX
from ..utils.errors import Edge2TorchError


def compute_hidden_nodes(
    *,
    node_names: list[str],
    input_nodes: list[str],
    output_nodes: list[str],
) -> list[str]:
    """
    Return sorted hidden graph nodes excluding inputs, outputs, and pseudo
    nodes.
    """
    input_set = set(input_nodes)
    output_set = set(output_nodes)

    return sorted(
        node_name
        for node_name in node_names
        if node_name not in input_set
        and node_name not in output_set
        and not node_name.startswith(PSEUDO_NODE_PREFIX)
    )


def build_feedforward_interpretation_sites(
    node_names_by_layer: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Build feedforward interpretation sites for non-input layers.
    """
    interpretation_sites: dict[str, list[str]] = {}

    for layer_name in sorted(
        node_names_by_layer.keys(),
        key=_feedforward_layer_sort_key,
    ):
        if layer_name == "layer_0":
            continue

        interpretation_sites[layer_name] = list(node_names_by_layer[layer_name])

    return interpretation_sites


def build_state_update_interpretation_sites(
    *,
    node_names: list[str],
    steps: int,
) -> dict[str, list[str]]:
    """
    Build recurrent or graphnn interpretation sites for each update step.
    """
    if steps <= 0:
        raise Edge2TorchError("'steps' must be a positive integer.")

    ordered_node_names = list(node_names)

    return {
        f"step_{step_idx}": list(ordered_node_names)
        for step_idx in range(1, steps + 1)
    }


def collect_feedforward_node_names(
    node_names_by_layer: dict[str, list[str]],
) -> list[str]:
    """
    Collect all node names referenced across feedforward layers.
    """
    return sorted(
        {
            node_name
            for layer_nodes in node_names_by_layer.values()
            for node_name in layer_nodes
        }
    )


def _feedforward_layer_sort_key(layer_name: str) -> int:
    """
    Sort feedforward layer names like ``layer_0``, ``layer_1``, ...
    """
    if not layer_name.startswith("layer_"):
        raise Edge2TorchError(f"Invalid layer name '{layer_name}'.")

    try:
        return int(layer_name.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise Edge2TorchError(f"Invalid layer name '{layer_name}'.") from exc
