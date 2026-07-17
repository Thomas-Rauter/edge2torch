"""
Execution-plan definitions and builders for compiled backends.

Why this file exists
--------------------
This file defines the intermediate execution-plan objects used to
translate a validated graph into backend-specific PyTorch models. The
execution plan exists to separate graph-structural reasoning from model
construction, so that compilation can first decide how a backend should
execute the graph and only then build the corresponding model.

Role in the package
-------------------
This is an internal compilation-structure module. It contains the
backend-specific execution-plan schemas and the logic that derives them
from an internal graph object. It should focus on execution structure,
not on public API orchestration, input validation, or PyTorch module
implementation.
"""

from dataclasses import dataclass

import pandas as pd

from ..graph.schema import EdgeGraph
from ..utils.constants import PSEUDO_NODE_PREFIX
from ..utils.errors import Edge2TorchError


@dataclass
class FeedforwardExecutionPlan:
    """
    Execution plan for a feedforward KPNN model.
    """

    original_edges: pd.DataFrame
    expanded_edges: pd.DataFrame
    node_names_by_layer: dict[str, list[str]]
    node_to_layer: dict[str, str]
    pseudo_nodes: list[str]
    input_node_names: list[str]
    output_node_names: list[str]


def build_feedforward_execution_plan(
    graph: EdgeGraph,
) -> FeedforwardExecutionPlan:
    """
    Build a feedforward execution plan from a KPNN graph.

    Skipped edges are expanded into chains of pseudo nodes so that the final
    expanded graph can be computed strictly layer by layer.

    Parameters
    ----------
    graph
        Internal KPNN graph object.

    Returns
    -------
    FeedforwardExecutionPlan
        Execution plan for feedforward compilation.

    Raises
    ------
    Edge2TorchError
        If the graph contains cycles or cannot be layered.
    """
    original_edges = graph.edges.copy()

    in_degree: dict[str, int] = {node: 0 for node in graph.nodes}
    children: dict[str, list[str]] = {node: [] for node in graph.nodes}
    parents: dict[str, list[str]] = {node: [] for node in graph.nodes}

    for row in original_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        in_degree[target] += 1
        children[source].append(target)
        parents[target].append(source)

    remaining = set(graph.nodes)
    current_layer_nodes = sorted(
        node for node in graph.nodes if in_degree[node] == 0
    )

    if not current_layer_nodes:
        raise Edge2TorchError(
            "Feedforward compilation requires at least one input node."
        )

    node_to_depth: dict[str, int] = {}
    depth = 0

    while current_layer_nodes:
        next_layer_candidates: set[str] = set()

        for node in current_layer_nodes:
            node_to_depth[node] = depth
            remaining.discard(node)

            for child in children[node]:
                in_degree[child] -= 1

                if in_degree[child] == 0:
                    next_layer_candidates.add(child)

        depth += 1
        current_layer_nodes = sorted(next_layer_candidates)

    if remaining:
        raise Edge2TorchError(
            "Feedforward compilation failed because the graph contains "
            "cycles or cannot be layered."
        )

    node_names_by_layer: dict[str, list[str]] = {}
    node_to_layer: dict[str, str] = {}

    max_depth = max(node_to_depth.values())

    for layer_idx in range(max_depth + 1):
        layer_name = f"layer_{layer_idx}"
        layer_nodes = sorted(
            node
            for node, node_depth in node_to_depth.items()
            if node_depth == layer_idx
        )

        node_names_by_layer[layer_name] = layer_nodes

        for node in layer_nodes:
            node_to_layer[node] = layer_name

    input_node_names = sorted(
        node for node in graph.nodes if len(parents[node]) == 0
    )

    output_node_names = sorted(
        node for node in graph.nodes if len(children[node]) == 0
    )

    output_node_depths = {
        node: node_to_depth[node] for node in output_node_names
    }

    if output_node_depths:
        final_output_depth = max(output_node_depths.values())
        early_output_nodes = sorted(
            node
            for node, node_depth in output_node_depths.items()
            if node_depth != final_output_depth
        )

        if early_output_nodes:
            early_output_str = ", ".join(early_output_nodes)
            raise Edge2TorchError(
                "Feedforward compilation requires all terminal output nodes "
                "to be at the same layer depth. Output node(s) at earlier "
                "depth than the final output layer are not supported: "
                f"{early_output_str}."
            )

    has_initial_weight = "initial_weight" in original_edges.columns
    has_constraint = "constraint" in original_edges.columns

    expanded_edge_columns = ["source", "target"]

    if has_initial_weight:
        expanded_edge_columns.append("initial_weight")

    if has_constraint:
        expanded_edge_columns.append("constraint")

    expanded_edges_records: list[dict[str, object]] = []
    pseudo_nodes: list[str] = []

    for row in original_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        source_depth = node_to_depth[source]
        target_depth = node_to_depth[target]
        depth_gap = target_depth - source_depth

        if depth_gap <= 0:
            raise Edge2TorchError(
                "Feedforward compilation requires all edges to point from "
                "earlier to later layers."
            )

        edge_metadata: dict[str, object] = {}

        if has_initial_weight:
            edge_metadata["initial_weight"] = getattr(row, "initial_weight")

        if has_constraint:
            edge_metadata["constraint"] = getattr(row, "constraint")

        if depth_gap == 1:
            expanded_edges_records.append(
                {
                    "source": source,
                    "target": target,
                    **edge_metadata,
                }
            )
            continue

        previous_node = source

        for pseudo_depth in range(source_depth + 1, target_depth):
            pseudo_node = (
                f"{PSEUDO_NODE_PREFIX}{source}__{target}__layer_{pseudo_depth}"
            )
            pseudo_nodes.append(pseudo_node)

            layer_name = f"layer_{pseudo_depth}"
            node_names_by_layer[layer_name].append(pseudo_node)
            node_to_layer[pseudo_node] = layer_name

            pseudo_edge_record: dict[str, object] = {
                "source": previous_node,
                "target": pseudo_node,
            }

            if has_initial_weight:
                pseudo_edge_record["initial_weight"] = float("nan")

            if has_constraint:
                pseudo_edge_record["constraint"] = "unconstrained"

            expanded_edges_records.append(pseudo_edge_record)

            previous_node = pseudo_node

        expanded_edges_records.append(
            {
                "source": previous_node,
                "target": target,
                **edge_metadata,
            }
        )

    for layer_name, layer_nodes in node_names_by_layer.items():
        node_names_by_layer[layer_name] = sorted(layer_nodes)

    expanded_edges = pd.DataFrame(
        expanded_edges_records,
        columns=expanded_edge_columns,
    )

    return FeedforwardExecutionPlan(
        original_edges=original_edges,
        expanded_edges=expanded_edges,
        node_names_by_layer=node_names_by_layer,
        node_to_layer=node_to_layer,
        pseudo_nodes=sorted(pseudo_nodes),
        input_node_names=input_node_names,
        output_node_names=output_node_names,
    )


@dataclass
class StateUpdateExecutionPlan:
    """
    Execution plan for topology-preserving state-update models.

    Used by the state_update backends, which currently share the
    same non-layerized execution structure.

    Attributes
    ----------
    original_edges : pd.DataFrame
        Normalized edge table used for state-update compilation. Always
        contains ``source`` and ``target`` columns. May also contain optional
        edge-level metadata columns such as ``initial_weight`` and
        ``constraint``.
    node_names : list[str]
        Names of all graph nodes in model state order.
    input_node_names : list[str]
        Names of input nodes inferred as nodes with no incoming edges.
    output_node_names : list[str]
        Names of output nodes inferred as nodes with no outgoing edges.
    """

    original_edges: pd.DataFrame
    node_names: list[str]
    input_node_names: list[str]
    output_node_names: list[str]


def build_state_update_execution_plan(
    graph: EdgeGraph,
    *,
    backend_label: str = "State-update",
) -> StateUpdateExecutionPlan:
    """
    Build a state-update execution plan from an edge2torch graph.

    The graph is kept in its original form rather than expanded into
    adjacent feedforward layers. Cycles are allowed, but the graph must
    expose at least one input node and one output node.

    Edge-level metadata stored in ``graph.edges``, such as
    ``initial_weight`` and ``constraint``, is preserved in
    ``original_edges`` and consumed later by the model constructor.

    Parameters
    ----------
    graph
        Internal edge2torch graph object.
    backend_label
        Label used in validation error messages.

    Returns
    -------
    StateUpdateExecutionPlan
        Execution plan for topology-preserving state-update compilation.

    Raises
    ------
    Edge2TorchError
        If the graph is empty or structurally invalid for state-update
        compilation.
    """
    original_edges = graph.edges.copy()

    if original_edges.empty:
        raise Edge2TorchError(
            f"{backend_label} compilation requires at least one edge."
        )

    node_names = list(graph.nodes)

    if not node_names:
        raise Edge2TorchError(
            f"{backend_label} compilation requires at least one node."
        )

    children: dict[str, list[str]] = {node: [] for node in node_names}
    parents: dict[str, list[str]] = {node: [] for node in node_names}

    for row in original_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        if source not in children:
            raise Edge2TorchError(
                f"Unknown source node '{source}' in "
                f"{backend_label.lower()} graph."
            )

        if target not in parents:
            raise Edge2TorchError(
                f"Unknown target node '{target}' in "
                f"{backend_label.lower()} graph."
            )

        children[source].append(target)
        parents[target].append(source)

    input_node_names = sorted(
        node for node in node_names if len(parents[node]) == 0
    )

    output_node_names = sorted(
        node for node in node_names if len(children[node]) == 0
    )

    if not input_node_names:
        raise Edge2TorchError(
            f"{backend_label} compilation requires at least one "
            "input node. Cycles are allowed, but the graph must "
            "include at least one node with no incoming edges."
        )

    if not output_node_names:
        raise Edge2TorchError(
            f"{backend_label} compilation requires at least one "
            "output node. Cycles are allowed, but the graph must "
            "include at least one node with no outgoing edges."
        )

    _validate_outputs_reachable_from_inputs(
        children=children,
        parents=parents,
        input_node_names=input_node_names,
        output_node_names=output_node_names,
        backend_name=backend_label,
    )

    return StateUpdateExecutionPlan(
        original_edges=original_edges,
        node_names=sorted(node_names),
        input_node_names=input_node_names,
        output_node_names=output_node_names,
    )


# Level 5 functions (functions called by level 4 functions) --------------------


def _validate_outputs_reachable_from_inputs(
    children: dict[str, list[str]],
    parents: dict[str, list[str]],
    input_node_names: list[str],
    output_node_names: list[str],
    backend_name: str,
) -> None:
    """
    Validate that every output-relevant node is reachable from an input node.
    """
    reachable_from_inputs: set[str] = set()
    stack = list(input_node_names)

    while stack:
        node = stack.pop()

        if node in reachable_from_inputs:
            continue

        reachable_from_inputs.add(node)
        stack.extend(children[node])

    output_relevant_nodes: set[str] = set()
    stack = list(output_node_names)

    while stack:
        node = stack.pop()

        if node in output_relevant_nodes:
            continue

        output_relevant_nodes.add(node)
        stack.extend(parents[node])

    unreachable_outputs = sorted(
        node for node in output_node_names if node not in reachable_from_inputs
    )

    if unreachable_outputs:
        unreachable_str = ", ".join(unreachable_outputs)
        raise Edge2TorchError(
            f"{backend_name} compilation requires every output node "
            "to be reachable from at least one input node. "
            f"Unreachable output node(s): {unreachable_str}."
        )

    unreachable_output_relevant_nodes = sorted(
        node
        for node in output_relevant_nodes
        if node not in reachable_from_inputs and node not in output_node_names
    )

    if unreachable_output_relevant_nodes:
        unreachable_str = ", ".join(unreachable_output_relevant_nodes)
        raise Edge2TorchError(
            f"{backend_name} compilation requires every node that can "
            "influence an output node to be reachable from at least one "
            "input node. Unreachable output-relevant node(s): "
            f"{unreachable_str}."
        )
