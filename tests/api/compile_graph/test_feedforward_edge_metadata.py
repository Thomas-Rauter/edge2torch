import math

import pandas as pd
import pytest
import torch

from edge2torch import compile_graph
from edge2torch.nn.masked_linear import (
    CONSTRAINT_FIXED,
    CONSTRAINT_NEGATIVE,
    CONSTRAINT_POSITIVE,
    CONSTRAINT_UNCONSTRAINED,
    ConstrainedMaskedLinear,
    MaskedLinear,
)


def _reset_if_available(module: torch.nn.Module) -> None:
    reset_parameters = getattr(module, "reset_parameters", None)

    if callable(reset_parameters):
        reset_parameters()


def _block_for_layer(model: torch.nn.Module, layer_name: str):
    return model._edge2torch_get_interpretation_site(layer_name)


def _edge_weight(
    block,
    source: str,
    target: str,
) -> float:
    source_idx = block.input_node_names.index(source)
    target_idx = block.output_node_names.index(target)

    return float(
        block.linear.effective_weight[target_idx, source_idx].detach().cpu()
    )


def _edge_constraint_code(
    block,
    source: str,
    target: str,
) -> int:
    source_idx = block.input_node_names.index(source)
    target_idx = block.output_node_names.index(target)

    return int(block.linear.constraint[target_idx, source_idx])


def _metadata_edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [
                "feature_a",
                "feature_b",
                "hidden_pos",
                "hidden_neg",
            ],
            "target": [
                "hidden_pos",
                "hidden_neg",
                "prediction",
                "prediction",
            ],
            "initial_weight": [
                0.25,
                -0.50,
                0.75,
                math.nan,
            ],
            "constraint": [
                "positive",
                "negative",
                "fixed",
                None,
            ],
        }
    )


def _skip_edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [
                "feature_a",
                "feature_a",
                "hidden",
                "middle",
            ],
            "target": [
                "hidden",
                "prediction",
                "middle",
                "prediction",
            ],
            "initial_weight": [
                0.20,
                0.75,
                0.30,
                -0.40,
            ],
            "constraint": [
                "positive",
                "fixed",
                "positive",
                "negative",
            ],
        }
    )


def test_feedforward_uses_constrained_layer_with_edge_metadata():
    edgelist = _metadata_edgelist()

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    assert artifact.backend == "feedforward"
    assert artifact.feature_names == ["feature_a", "feature_b"]

    for block in model.blocks:
        assert isinstance(block.linear, ConstrainedMaskedLinear)

    assert list(artifact.graph.edges.columns) == [
        "source",
        "target",
        "initial_weight",
        "constraint",
    ]

    assert artifact.graph.edges["constraint"].tolist() == [
        "positive",
        "negative",
        "fixed",
        "unconstrained",
    ]


def test_feedforward_initializes_weights_and_constraints():
    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    first_block = _block_for_layer(model, "layer_1")
    second_block = _block_for_layer(model, "layer_2")

    assert (
        _edge_constraint_code(
            first_block,
            source="feature_a",
            target="hidden_pos",
        )
        == CONSTRAINT_POSITIVE
    )
    assert (
        _edge_constraint_code(
            first_block,
            source="feature_b",
            target="hidden_neg",
        )
        == CONSTRAINT_NEGATIVE
    )
    assert (
        _edge_constraint_code(
            second_block,
            source="hidden_pos",
            target="prediction",
        )
        == CONSTRAINT_FIXED
    )
    assert (
        _edge_constraint_code(
            second_block,
            source="hidden_neg",
            target="prediction",
        )
        == CONSTRAINT_UNCONSTRAINED
    )

    assert _edge_weight(
        first_block,
        source="feature_a",
        target="hidden_pos",
    ) == pytest.approx(0.25)
    assert _edge_weight(
        first_block,
        source="feature_b",
        target="hidden_neg",
    ) == pytest.approx(-0.50)
    assert _edge_weight(
        second_block,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)

    unconstrained_weight = _edge_weight(
        second_block,
        source="hidden_neg",
        target="prediction",
    )
    assert math.isfinite(unconstrained_weight)


def test_feedforward_without_edge_metadata_uses_masked_linear():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "feature_b", "hidden"],
            "target": ["hidden", "hidden", "prediction"],
        }
    )

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    for block in model.blocks:
        assert isinstance(block.linear, MaskedLinear)
        assert not isinstance(block.linear, ConstrainedMaskedLinear)


def test_feedforward_skip_edge_metadata_reaches_final_edge():
    edgelist = _skip_edgelist()

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    pseudo_nodes = artifact.execution_plan.pseudo_nodes

    assert len(pseudo_nodes) == 2

    first_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_1"
    final_pseudo = "__edge2torch_pseudo__feature_a__prediction__layer_2"

    assert first_pseudo in pseudo_nodes
    assert final_pseudo in pseudo_nodes

    expanded_edges = artifact.execution_plan.expanded_edges

    first_pseudo_edge = expanded_edges[
        (expanded_edges["source"] == "feature_a")
        & (expanded_edges["target"] == first_pseudo)
    ].iloc[0]

    second_pseudo_edge = expanded_edges[
        (expanded_edges["source"] == first_pseudo)
        & (expanded_edges["target"] == final_pseudo)
    ].iloc[0]

    final_skip_edge = expanded_edges[
        (expanded_edges["source"] == final_pseudo)
        & (expanded_edges["target"] == "prediction")
    ].iloc[0]

    assert pd.isna(first_pseudo_edge["initial_weight"])
    assert first_pseudo_edge["constraint"] == "unconstrained"

    assert pd.isna(second_pseudo_edge["initial_weight"])
    assert second_pseudo_edge["constraint"] == "unconstrained"

    assert final_skip_edge["initial_weight"] == pytest.approx(0.75)
    assert final_skip_edge["constraint"] == "fixed"

    final_block = _block_for_layer(model, "layer_3")

    assert (
        _edge_constraint_code(
            final_block,
            source=final_pseudo,
            target="prediction",
        )
        == CONSTRAINT_FIXED
    )
    assert _edge_weight(
        final_block,
        source=final_pseudo,
        target="prediction",
    ) == pytest.approx(0.75)


def test_feedforward_pseudo_node_still_passes_input_through():
    edgelist = _skip_edgelist()

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    pseudo_node = artifact.execution_plan.pseudo_nodes[0]
    first_block = _block_for_layer(model, "layer_1")

    x = torch.tensor([[2.0]])

    with torch.no_grad():
        y = first_block(x)

    pseudo_idx = first_block.output_node_names.index(pseudo_node)

    assert float(y[0, pseudo_idx]) == pytest.approx(2.0)


def test_feedforward_constraints_survive_optimizer_steps():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    first_block = _block_for_layer(model, "layer_1")
    second_block = _block_for_layer(model, "layer_2")

    fixed_before = _edge_weight(
        second_block,
        source="hidden_pos",
        target="prediction",
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.tensor(
        [
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )

    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()

        assert (
            _edge_weight(
                first_block,
                source="feature_a",
                target="hidden_pos",
            )
            > 0
        )
        assert (
            _edge_weight(
                first_block,
                source="feature_b",
                target="hidden_neg",
            )
            < 0
        )
        assert _edge_weight(
            second_block,
            source="hidden_pos",
            target="prediction",
        ) == pytest.approx(fixed_before)


def test_feedforward_fixed_weights_are_buffers():
    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    assert "blocks.0.linear.raw_weight" in parameter_names
    assert "blocks.1.linear.raw_weight" in parameter_names

    assert "blocks.0.linear.fixed_weight" not in parameter_names
    assert "blocks.1.linear.fixed_weight" not in parameter_names

    assert "blocks.0.linear.fixed_weight" in buffer_names
    assert "blocks.0.linear.constraint" in buffer_names
    assert "blocks.0.linear.initial_effective_weight" in buffer_names
    assert "blocks.0.linear.mask" in buffer_names

    assert "blocks.1.linear.fixed_weight" in buffer_names
    assert "blocks.1.linear.constraint" in buffer_names
    assert "blocks.1.linear.initial_effective_weight" in buffer_names
    assert "blocks.1.linear.mask" in buffer_names


def test_feedforward_reset_restores_explicit_initial_weights():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])

    optimizer.zero_grad()
    loss = model(x).pow(2).sum()
    loss.backward()
    optimizer.step()

    model.apply(_reset_if_available)

    first_block = _block_for_layer(model, "layer_1")
    second_block = _block_for_layer(model, "layer_2")

    assert _edge_weight(
        first_block,
        source="feature_a",
        target="hidden_pos",
    ) == pytest.approx(0.25)
    assert _edge_weight(
        first_block,
        source="feature_b",
        target="hidden_neg",
    ) == pytest.approx(-0.50)
    assert _edge_weight(
        second_block,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)


def test_feedforward_state_dict_roundtrip_preserves_behavior():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])

    optimizer.zero_grad()
    loss = model(x).pow(2).sum()
    loss.backward()
    optimizer.step()

    state_dict = model.state_dict()

    reloaded_model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
    )
    reloaded_model.load_state_dict(state_dict)

    first_block = _block_for_layer(reloaded_model, "layer_1")
    second_block = _block_for_layer(reloaded_model, "layer_2")

    assert (
        _edge_constraint_code(
            first_block,
            source="feature_a",
            target="hidden_pos",
        )
        == CONSTRAINT_POSITIVE
    )
    assert (
        _edge_constraint_code(
            first_block,
            source="feature_b",
            target="hidden_neg",
        )
        == CONSTRAINT_NEGATIVE
    )
    assert (
        _edge_constraint_code(
            second_block,
            source="hidden_pos",
            target="prediction",
        )
        == CONSTRAINT_FIXED
    )

    assert (
        _edge_weight(
            first_block,
            source="feature_a",
            target="hidden_pos",
        )
        > 0
    )
    assert (
        _edge_weight(
            first_block,
            source="feature_b",
            target="hidden_neg",
        )
        < 0
    )
    assert _edge_weight(
        second_block,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)

    torch.testing.assert_close(model(x), reloaded_model(x))
