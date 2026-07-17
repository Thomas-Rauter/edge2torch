"""
Site-wise node-level Captum attribution methods.

Why this file exists
--------------------
This file contains Captum method implementations that attribute model outputs
to named graph nodes at interpretation sites. Site attribution is implemented
via Captum layer-level methods and then mapped back to named graph nodes.

Role in the package
-------------------
This is an internal interpretation module. It should implement site-wise and
summary node attribution across all compiled backends. It should not handle
public API validation, graph compilation, input preparation, or plotting.
"""

from collections.abc import Callable
from typing import Any, Literal, Union, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..compile.artifact import CompileArtifact
from ..nn.interpretation_sites import find_interpretation_site_provider
from ..utils.constants import INTERNAL_NODE_PREFIX, PSEUDO_NODE_PREFIX
from ..utils.errors import Edge2TorchError
from .captum_classes import get_captum_class
from .method_registry import (
    NODE_METHODS_WITH_CONSTRUCTOR_KWARGS,
    NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS,
)

NodeFilter = Literal["hidden", "all", "non_input"]
InterpretationLevel = Literal["sites", "summary"]
SiteAggregation = Literal["max_abs", "mean_abs", "last"]

# Level 2 functions (called by level 1 functions) ------------------------------


def run_site_node_attribution(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
    *,
    nodes: NodeFilter = "hidden",
    level: InterpretationLevel = "summary",
    site_aggregation: SiteAggregation = "max_abs",
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run node-level attribution at interpretation sites for any backend.
    """
    site_results = _run_site_wise_attribution(
        model=model,
        artifact=artifact,
        inputs=inputs,
        sample_names=sample_names,
        method=method,
        constructor_kwargs=constructor_kwargs,
        attribute_kwargs=attribute_kwargs,
        nodes=nodes,
    )

    if level == "sites":
        return site_results

    if level == "summary":
        return _build_summary_attribution(
            artifact=artifact,
            site_results=site_results,
            site_aggregation=site_aggregation,
        )

    raise Edge2TorchError(
        f"Unsupported interpretation level '{level}' for target='nodes'."
    )


# Level 3 functions (called by level 2 functions) ------------------------------


def _run_site_wise_attribution(
    model: nn.Module,
    artifact: CompileArtifact,
    inputs: torch.Tensor,
    sample_names: list[str],
    method: str,
    constructor_kwargs: dict[str, Any],
    attribute_kwargs: dict[str, Any],
    nodes: NodeFilter,
) -> dict[str, pd.DataFrame]:
    """
    Run node-level attribution and return one DataFrame per site.
    """
    results: dict[str, pd.DataFrame] = {}

    site_provider = find_interpretation_site_provider(model)
    get_site = cast(
        Callable[[str], nn.Module],
        getattr(site_provider, "_edge2torch_get_interpretation_site"),
    )

    for site_id in sorted(
        artifact.interpretation_sites.keys(),
        key=_site_sort_key,
    ):
        site_node_names = artifact.interpretation_sites[site_id]
        site_module = get_site(site_id)

        interpreter = _build_site_interpreter(
            method=method,
            model=model,
            site_module=site_module,
            constructor_kwargs=constructor_kwargs,
        )

        interpreter = cast(Any, interpreter)
        attributions = interpreter.attribute(
            inputs,
            **attribute_kwargs,
        )

        _validate_node_attributions(
            attributions=attributions,
            site_id=site_id,
            sample_names=sample_names,
            node_names=site_node_names,
        )

        filtered_node_names, filtered_attributions = _filter_site_attributions(
            attributions=attributions,
            node_names=site_node_names,
            artifact=artifact,
            nodes=nodes,
        )

        if not filtered_node_names:
            continue

        results[site_id] = pd.DataFrame(
            filtered_attributions.detach().cpu().numpy(),
            index=sample_names,
            columns=filtered_node_names,
        )

    if not results:
        raise Edge2TorchError(
            "Node interpretation produced no visible nodes for the "
            f"requested node filter '{nodes}'."
        )

    return results


def _build_summary_attribution(
    artifact: CompileArtifact,
    site_results: dict[str, pd.DataFrame],
    site_aggregation: SiteAggregation,
) -> pd.DataFrame:
    """
    Build one node-importance table from per-site attribution results.
    """
    if not site_results:
        raise Edge2TorchError(
            "Cannot build summary node attribution from empty site results."
        )

    if artifact.backend == "feedforward":
        return _merge_feedforward_site_attributions(site_results)

    if artifact.backend == "state_update":
        return _aggregate_state_update_site_attributions(
            site_results=site_results,
            site_aggregation=site_aggregation,
        )

    raise Edge2TorchError(
        f"Unsupported backend '{artifact.backend}' for summary node "
        "attribution."
    )


def _merge_feedforward_site_attributions(
    site_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge disjoint feedforward site tables into one summary DataFrame.
    """
    site_ids = sorted(site_results.keys(), key=_site_sort_key)
    summary = site_results[site_ids[0]]

    for site_id in site_ids[1:]:
        summary = pd.concat([summary, site_results[site_id]], axis=1)

    return summary


def _aggregate_state_update_site_attributions(
    site_results: dict[str, pd.DataFrame],
    site_aggregation: SiteAggregation,
) -> pd.DataFrame:
    """
    Aggregate repeated node columns across state-update steps.
    """
    site_ids = sorted(site_results.keys(), key=_site_sort_key)

    if site_aggregation == "last":
        return site_results[site_ids[-1]].copy()

    arrays = np.stack(
        [site_results[site_id].to_numpy() for site_id in site_ids],
        axis=0,
    )

    if site_aggregation == "max_abs":
        abs_arrays = np.abs(arrays)
        max_indices = abs_arrays.argmax(axis=0)
        n_sites, n_samples, n_nodes = arrays.shape

        row_indices = np.arange(n_samples)[:, None]
        col_indices = np.arange(n_nodes)[None, :]
        summary_values = arrays[max_indices, row_indices, col_indices]

        return pd.DataFrame(
            summary_values,
            index=site_results[site_ids[0]].index,
            columns=site_results[site_ids[0]].columns,
        )

    if site_aggregation == "mean_abs":
        summary_values = np.mean(np.abs(arrays), axis=0)

        return pd.DataFrame(
            summary_values,
            index=site_results[site_ids[0]].index,
            columns=site_results[site_ids[0]].columns,
        )

    raise Edge2TorchError(f"Unsupported site aggregation '{site_aggregation}'.")


def _filter_site_attributions(
    attributions: torch.Tensor,
    node_names: list[str],
    artifact: CompileArtifact,
    nodes: NodeFilter,
) -> tuple[list[str], torch.Tensor]:
    """
    Filter site attributions to the requested node subset.
    """
    selected_indices = [
        idx
        for idx, node_name in enumerate(node_names)
        if _should_include_node(
            node_name=node_name,
            artifact=artifact,
            nodes=nodes,
        )
    ]

    if not selected_indices:
        return [], attributions[:, :0]

    selected_node_names = [node_names[idx] for idx in selected_indices]
    selected_attributions = attributions[:, selected_indices]

    return selected_node_names, selected_attributions


def _should_include_node(
    node_name: str,
    artifact: CompileArtifact,
    nodes: NodeFilter,
) -> bool:
    """
    Return True if a node should be included for the requested node filter.
    """
    if not _is_visible_domain_node(node_name):
        return False

    if nodes == "all":
        return True

    if nodes == "hidden":
        return node_name in set(artifact.hidden_nodes)

    if nodes == "non_input":
        return node_name not in set(artifact.input_nodes)

    raise Edge2TorchError(f"Unsupported node filter '{nodes}'.")


def _build_site_interpreter(
    method: str,
    model: nn.Module,
    site_module: nn.Module,
    constructor_kwargs: dict[str, Any],
):
    """
    Build a Captum interpreter for site-level attribution.
    """
    if method in NODE_METHODS_WITH_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(
            model,
            site_module,
            **constructor_kwargs,
        )

    if method in NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS:
        interpreter_class = get_captum_class(method)
        return interpreter_class(
            model,
            site_module,
        )

    raise Edge2TorchError(
        f"Method '{method}' is not supported for target='nodes'."
    )


def _validate_node_attributions(
    attributions: torch.Tensor,
    site_id: str,
    sample_names: list[str],
    node_names: list[str],
) -> None:
    """
    Validate node-attribution output shape for one site.
    """
    if attributions.ndim != 2:
        raise Edge2TorchError(
            f"Node attributions for '{site_id}' must have "
            "shape (n_examples, n_nodes)."
        )

    n_examples, n_nodes = attributions.shape

    if n_examples != len(sample_names):
        raise Edge2TorchError(
            f"Node attribution row count for '{site_id}' "
            "does not match sample count."
        )

    if n_nodes != len(node_names):
        raise Edge2TorchError(
            f"Node attribution width mismatch for '{site_id}'. "
            f"Expected {len(node_names)}, got {n_nodes}."
        )


def _is_visible_domain_node(node_name: str) -> bool:
    """
    Return True if a node should be exposed in interpretation output.
    """
    return not (
        node_name.startswith(INTERNAL_NODE_PREFIX)
        or node_name.startswith(PSEUDO_NODE_PREFIX)
    )


def _site_sort_key(site_id: str) -> tuple[int, int]:
    """
    Sort interpretation site identifiers from any backend.
    """
    if site_id.startswith("layer_"):
        try:
            return (0, int(site_id.split("_")[1]))
        except (IndexError, ValueError) as exc:
            raise Edge2TorchError(
                f"Invalid interpretation site '{site_id}' in artifact."
            ) from exc

    if site_id.startswith("step_"):
        try:
            return (1, int(site_id.split("_")[1]))
        except (IndexError, ValueError) as exc:
            raise Edge2TorchError(
                f"Invalid interpretation site '{site_id}' in artifact."
            ) from exc

    raise Edge2TorchError(
        f"Invalid interpretation site '{site_id}' in artifact."
    )
