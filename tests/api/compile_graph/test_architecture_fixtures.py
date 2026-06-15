from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import torch

from edge2torch.compile_graph import compile_graph
from edge2torch.nn.masked_linear import MaskedLinear
from edge2torch.nn.model import EdgeGraphNNModel, EdgeModel, RecurrentEdgeModel
from edge2torch.utils.constants import PSEUDO_NODE_PREFIX

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "edgelists"


def _load_edgelist(filename: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / filename)


def _edge_set(edges: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.source), str(row.target))
        for row in edges.itertuples(index=False)
    }


def _source_nodes(edgelist: pd.DataFrame) -> set[str]:
    return set(edgelist["source"])


def _target_nodes(edgelist: pd.DataFrame) -> set[str]:
    return set(edgelist["target"])


def _all_nodes(edgelist: pd.DataFrame) -> list[str]:
    return sorted(_source_nodes(edgelist).union(_target_nodes(edgelist)))


def _input_nodes(edgelist: pd.DataFrame) -> list[str]:
    return sorted(_source_nodes(edgelist).difference(_target_nodes(edgelist)))


def _output_nodes(edgelist: pd.DataFrame) -> list[str]:
    return sorted(_target_nodes(edgelist).difference(_source_nodes(edgelist)))


def _is_pseudo_node(node_name: str) -> bool:
    return node_name.startswith(PSEUDO_NODE_PREFIX)


def _layer_index(layer_name: str) -> int:
    return int(layer_name.split("_")[1])


def _node_depths(node_to_layer: dict[str, str]) -> dict[str, int]:
    return {
        node_name: _layer_index(layer_name)
        for node_name, layer_name in node_to_layer.items()
    }


def _mask_from_masked_linear(module: MaskedLinear) -> torch.Tensor:
    return module.mask.detach().cpu()


def _masked_linear_from_feedforward_block(block) -> MaskedLinear:
    masked_linear_modules = [
        module for module in block.modules() if isinstance(module, MaskedLinear)
    ]

    assert len(masked_linear_modules) == 1

    return masked_linear_modules[0]


def _expected_mask(
    *,
    input_node_names: list[str],
    output_node_names: list[str],
    edges: pd.DataFrame,
) -> torch.Tensor:
    input_index = {
        node_name: idx for idx, node_name in enumerate(input_node_names)
    }
    output_index = {
        node_name: idx for idx, node_name in enumerate(output_node_names)
    }

    mask = torch.zeros(
        len(output_node_names),
        len(input_node_names),
        dtype=torch.float32,
    )

    for row in edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        if source in input_index and target in output_index:
            mask[output_index[target], input_index[source]] = 1.0

    return mask


def _expected_state_update_mask(
    *,
    node_names: list[str],
    edges: pd.DataFrame,
) -> torch.Tensor:
    node_index = {node_name: idx for idx, node_name in enumerate(node_names)}

    mask = torch.zeros(
        len(node_names),
        len(node_names),
        dtype=torch.float32,
    )

    for row in edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        mask[node_index[target], node_index[source]] = 1.0

    return mask


def _assert_feedforward_block_mask_matches_expanded_edges(
    *,
    model: EdgeModel,
    layer_name: str,
) -> None:
    plan = model.execution_plan
    layer_idx = _layer_index(layer_name)

    input_layer_name = f"layer_{layer_idx - 1}"
    output_layer_name = layer_name

    input_node_names = plan.node_names_by_layer[input_layer_name]
    output_node_names = plan.node_names_by_layer[output_layer_name]

    block = model._edge2torch_get_interpretation_site(layer_name)
    masked_linear = _masked_linear_from_feedforward_block(block)

    expected_mask = _expected_mask(
        input_node_names=input_node_names,
        output_node_names=output_node_names,
        edges=plan.expanded_edges,
    )

    assert torch.equal(_mask_from_masked_linear(masked_linear), expected_mask)


def _children_by_source(edges: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)

    for row in edges.itertuples(index=False):
        children[str(row.source)].append(str(row.target))

    return dict(children)


def _all_simple_paths(
    *,
    edges: pd.DataFrame,
    source: str,
    target: str,
) -> list[list[str]]:
    children = _children_by_source(edges)
    paths: list[list[str]] = []
    queue: deque[list[str]] = deque([[source]])

    while queue:
        path = queue.popleft()
        current_node = path[-1]

        if current_node == target:
            paths.append(path)
            continue

        for child in children.get(current_node, []):
            if child in path:
                continue

            queue.append([*path, child])

    return paths


def _assert_original_feedforward_edge_is_represented_exactly_once(
    *,
    expanded_edges: pd.DataFrame,
    source: str,
    target: str,
) -> None:
    paths = _all_simple_paths(
        edges=expanded_edges,
        source=source,
        target=target,
    )

    edge_expansion_paths = [
        path
        for path in paths
        if all(_is_pseudo_node(node) for node in path[1:-1])
    ]

    assert len(edge_expansion_paths) == 1

    path = edge_expansion_paths[0]

    assert path[0] == source
    assert path[-1] == target


def _assert_no_unexpected_domain_edges_in_expanded_feedforward_graph(
    *,
    original_edges: pd.DataFrame,
    expanded_edges: pd.DataFrame,
) -> None:
    original_edge_set = _edge_set(original_edges)

    expanded_domain_edges = {
        (source, target)
        for source, target in _edge_set(expanded_edges)
        if not _is_pseudo_node(source) and not _is_pseudo_node(target)
    }

    assert expanded_domain_edges.issubset(original_edge_set)


def _assert_feedforward_expansion_preserves_original_graph(
    *,
    original_edges: pd.DataFrame,
    expanded_edges: pd.DataFrame,
) -> None:
    for row in original_edges.itertuples(index=False):
        _assert_original_feedforward_edge_is_represented_exactly_once(
            expanded_edges=expanded_edges,
            source=str(row.source),
            target=str(row.target),
        )

    _assert_no_unexpected_domain_edges_in_expanded_feedforward_graph(
        original_edges=original_edges,
        expanded_edges=expanded_edges,
    )


def _assert_feedforward_layers_are_adjacent(
    *,
    expanded_edges: pd.DataFrame,
    node_to_layer: dict[str, str],
) -> None:
    depths = _node_depths(node_to_layer)

    for row in expanded_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)

        assert depths[target] - depths[source] == 1


def _assert_feedforward_model_masks_preserve_expanded_graph(
    model: EdgeModel,
) -> None:
    for layer_name in model.layer_names[1:]:
        _assert_feedforward_block_mask_matches_expanded_edges(
            model=model,
            layer_name=layer_name,
        )


def _assert_recurrent_or_graphnn_model_mask_matches_edgelist(
    *,
    node_names: list[str],
    edgelist: pd.DataFrame,
    masked_linear: MaskedLinear,
) -> None:
    expected_mask = _expected_state_update_mask(
        node_names=node_names,
        edges=edgelist,
    )

    assert torch.equal(
        _mask_from_masked_linear(masked_linear),
        expected_mask,
    )


def test_compile_graph_fixture_feedforward_multilayer_preserves_graph():
    edgelist = _load_edgelist("feedforward_multilayer.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    assert isinstance(model, EdgeModel)
    assert model.backend == "feedforward"
    assert artifact.backend == "feedforward"

    assert artifact.feature_names == _input_nodes(edgelist)
    assert artifact.execution_plan.input_node_names == _input_nodes(edgelist)
    assert artifact.execution_plan.output_node_names == _output_nodes(edgelist)
    assert artifact.execution_plan.pseudo_nodes == []

    assert artifact.execution_plan.original_edges.equals(edgelist)
    assert _edge_set(artifact.execution_plan.expanded_edges) == _edge_set(
        edgelist
    )

    assert model.layer_names == sorted(
        artifact.node_names_by_layer,
        key=_layer_index,
    )
    assert len(model.blocks) == len(model.layer_names) - 1

    assert artifact.node_names_by_layer[model.layer_names[0]] == _input_nodes(
        edgelist
    )
    assert artifact.node_names_by_layer[model.layer_names[-1]] == _output_nodes(
        edgelist
    )

    _assert_feedforward_layers_are_adjacent(
        expanded_edges=artifact.execution_plan.expanded_edges,
        node_to_layer=artifact.execution_plan.node_to_layer,
    )

    _assert_feedforward_expansion_preserves_original_graph(
        original_edges=edgelist,
        expanded_edges=artifact.execution_plan.expanded_edges,
    )

    _assert_feedforward_model_masks_preserve_expanded_graph(model)

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert y.shape == (4, len(_output_nodes(edgelist)))


def test_compile_graph_fixture_feedforward_skip_edges_preserves_graph():
    edgelist = _load_edgelist("feedforward_skip_edges.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    assert isinstance(model, EdgeModel)
    assert model.backend == "feedforward"
    assert artifact.backend == "feedforward"

    assert artifact.feature_names == _input_nodes(edgelist)
    assert artifact.execution_plan.input_node_names == _input_nodes(edgelist)
    assert artifact.execution_plan.output_node_names == _output_nodes(edgelist)

    assert artifact.execution_plan.original_edges.equals(edgelist)

    pseudo_nodes = artifact.execution_plan.pseudo_nodes

    assert pseudo_nodes
    assert all(_is_pseudo_node(node) for node in pseudo_nodes)

    for pseudo_node in pseudo_nodes:
        assert pseudo_node in artifact.execution_plan.node_to_layer
        assert any(
            pseudo_node in layer_nodes
            for layer_nodes in artifact.node_names_by_layer.values()
        )

    assert len(artifact.execution_plan.expanded_edges) == (
        len(edgelist) + len(pseudo_nodes)
    )

    _assert_feedforward_layers_are_adjacent(
        expanded_edges=artifact.execution_plan.expanded_edges,
        node_to_layer=artifact.execution_plan.node_to_layer,
    )

    _assert_feedforward_expansion_preserves_original_graph(
        original_edges=edgelist,
        expanded_edges=artifact.execution_plan.expanded_edges,
    )

    _assert_feedforward_model_masks_preserve_expanded_graph(model)

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert y.shape == (4, len(_output_nodes(edgelist)))


def test_compile_graph_fixture_recurrent_cycle_preserves_graph():
    edgelist = _load_edgelist("recurrent_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="recurrent",
        quiet=True,
    )

    assert isinstance(model, RecurrentEdgeModel)
    assert model.backend == "recurrent"
    assert artifact.backend == "recurrent"

    assert artifact.execution_plan.original_edges.equals(edgelist)

    assert model.input_node_names == _input_nodes(edgelist)
    assert model.output_node_names == _output_nodes(edgelist)
    assert artifact.feature_names == model.input_node_names

    expected_node_names = _all_nodes(edgelist)

    assert model.node_names == expected_node_names
    assert artifact.execution_plan.node_names == expected_node_names
    assert artifact.execution_plan.input_node_names == model.input_node_names
    assert artifact.execution_plan.output_node_names == model.output_node_names

    _assert_recurrent_or_graphnn_model_mask_matches_edgelist(
        node_names=model.node_names,
        edgelist=edgelist,
        masked_linear=model.recurrent,
    )

    x = torch.randn(4, len(model.input_node_names))
    y = model(x)

    assert y.shape == (4, len(model.output_node_names))


def test_compile_graph_fixture_graphnn_cycle_preserves_graph():
    edgelist = _load_edgelist("graphnn_cycle.csv")

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    assert isinstance(model, EdgeGraphNNModel)
    assert model.backend == "graphnn"
    assert artifact.backend == "graphnn"

    assert artifact.execution_plan.original_edges.equals(edgelist)

    assert model.input_node_names == _input_nodes(edgelist)
    assert model.output_node_names == _output_nodes(edgelist)
    assert artifact.feature_names == model.input_node_names

    expected_node_names = _all_nodes(edgelist)

    assert model.node_names == expected_node_names
    assert artifact.execution_plan.node_names == expected_node_names
    assert artifact.execution_plan.input_node_names == model.input_node_names
    assert artifact.execution_plan.output_node_names == model.output_node_names

    _assert_recurrent_or_graphnn_model_mask_matches_edgelist(
        node_names=model.node_names,
        edgelist=edgelist,
        masked_linear=model.message_passing,
    )

    x = torch.randn(4, len(model.input_node_names))
    y = model(x)

    assert y.shape == (4, len(model.output_node_names))
