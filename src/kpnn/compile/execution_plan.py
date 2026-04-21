from dataclasses import dataclass

import pandas as pd

from ..utils.errors import KPNNError


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


def build_feedforward_execution_plan(graph) -> FeedforwardExecutionPlan:
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
    KPNNError
        If the graph contains cycles or cannot be layered.
    """
    original_edges = graph.edges.copy()

    in_degree = {node: 0 for node in graph.nodes}
    children = {node: [] for node in graph.nodes}
    parents = {node: [] for node in graph.nodes}

    for _, row in original_edges.iterrows():
        source = row["source"]
        target = row["target"]

        in_degree[target] += 1
        children[source].append(target)
        parents[target].append(source)

    remaining = set(graph.nodes)
    current_layer_nodes = sorted(
        [node for node in graph.nodes if in_degree[node] == 0]
    )

    if not current_layer_nodes:
        raise KPNNError(
            "Feedforward compilation requires at least one input node."
        )

    node_to_depth = {}
    depth = 0

    while current_layer_nodes:
        next_layer_candidates = set()

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
        raise KPNNError(
            "Feedforward compilation failed because the graph contains "
            "cycles or cannot be layered."
        )

    node_names_by_layer = {}
    node_to_layer = {}

    max_depth = max(node_to_depth.values())

    for layer_idx in range(max_depth + 1):
        layer_name = f"layer_{layer_idx}"
        layer_nodes = sorted(
            [
                node for node, node_depth in node_to_depth.items()
                if node_depth == layer_idx
            ]
        )

        node_names_by_layer[layer_name] = layer_nodes

        for node in layer_nodes:
            node_to_layer[node] = layer_name

    input_node_names = sorted(
        [node for node in graph.nodes if len(parents[node]) == 0]
    )

    output_node_names = sorted(
        [node for node in graph.nodes if len(children[node]) == 0]
    )

    expanded_edges_records = []
    pseudo_nodes = []

    for _, row in original_edges.iterrows():
        source = row["source"]
        target = row["target"]

        source_depth = node_to_depth[source]
        target_depth = node_to_depth[target]
        depth_gap = target_depth - source_depth

        if depth_gap <= 0:
            raise KPNNError(
                "Feedforward compilation requires all edges to point from "
                "earlier to later layers."
            )

        if depth_gap == 1:
            expanded_edges_records.append(
                {"source": source, "target": target}
            )
            continue

        previous_node = source

        for step in range(1, depth_gap):
            pseudo_depth = source_depth + step
            is_last_step = step == depth_gap - 1

            if is_last_step:
                next_node = target
            else:
                next_node = (
                    f"pseudo__{source}__{target}__layer_{pseudo_depth}"
                )
                pseudo_nodes.append(next_node)

                layer_name = f"layer_{pseudo_depth}"
                node_names_by_layer[layer_name].append(next_node)
                node_to_layer[next_node] = layer_name

            expanded_edges_records.append(
                {"source": previous_node, "target": next_node}
            )

            previous_node = next_node

    for layer_name, layer_nodes in node_names_by_layer.items():
        node_names_by_layer[layer_name] = sorted(layer_nodes)

    expanded_edges = pd.DataFrame(
        expanded_edges_records,
        columns=["source", "target"],
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
