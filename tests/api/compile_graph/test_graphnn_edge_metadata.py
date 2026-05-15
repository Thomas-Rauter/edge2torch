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


def _edge_weight(model: torch.nn.Module, source: str, target: str) -> float:
    source_idx = model.node_index[source]
    target_idx = model.node_index[target]
    return float(
        model.message_passing.effective_weight[target_idx, source_idx]
        .detach()
        .cpu()
    )


def _edge_constraint_code(
    model: torch.nn.Module,
    source: str,
    target: str,
) -> int:
    source_idx = model.node_index[source]
    target_idx = model.node_index[target]
    return int(model.message_passing.constraint[target_idx, source_idx])


def _reset_if_available(module: torch.nn.Module) -> None:
    reset_parameters = getattr(module, "reset_parameters", None)

    if callable(reset_parameters):
        reset_parameters()


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


def test_graphnn_uses_constrained_layer_with_edge_metadata():
    edgelist = _metadata_edgelist()

    model, artifact = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    assert isinstance(model.message_passing, ConstrainedMaskedLinear)
    assert artifact.backend == "graphnn"
    assert artifact.feature_names == ["feature_a", "feature_b"]

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


def test_graphnn_initializes_weights_and_constraints():
    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    assert _edge_constraint_code(
        model,
        source="feature_a",
        target="hidden_pos",
    ) == CONSTRAINT_POSITIVE
    assert _edge_constraint_code(
        model,
        source="feature_b",
        target="hidden_neg",
    ) == CONSTRAINT_NEGATIVE
    assert _edge_constraint_code(
        model,
        source="hidden_pos",
        target="prediction",
    ) == CONSTRAINT_FIXED
    assert _edge_constraint_code(
        model,
        source="hidden_neg",
        target="prediction",
    ) == CONSTRAINT_UNCONSTRAINED

    assert _edge_weight(
        model,
        source="feature_a",
        target="hidden_pos",
    ) == pytest.approx(0.25)
    assert _edge_weight(
        model,
        source="feature_b",
        target="hidden_neg",
    ) == pytest.approx(-0.50)
    assert _edge_weight(
        model,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)

    unconstrained_weight = _edge_weight(
        model,
        source="hidden_neg",
        target="prediction",
    )
    assert math.isfinite(unconstrained_weight)


def test_graphnn_without_edge_metadata_uses_masked_linear():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "feature_b", "hidden"],
            "target": ["hidden", "hidden", "prediction"],
        }
    )

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    assert isinstance(model.message_passing, MaskedLinear)
    assert not isinstance(model.message_passing, ConstrainedMaskedLinear)


def test_graphnn_constraints_survive_optimizer_steps():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    fixed_before = _edge_weight(
        model,
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

        assert _edge_weight(
            model,
            source="feature_a",
            target="hidden_pos",
        ) > 0
        assert _edge_weight(
            model,
            source="feature_b",
            target="hidden_neg",
        ) < 0
        assert _edge_weight(
            model,
            source="hidden_pos",
            target="prediction",
        ) == pytest.approx(fixed_before)


def test_graphnn_fixed_weights_are_buffers():
    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    assert "message_passing.raw_weight" in parameter_names
    assert "message_passing.fixed_weight" not in parameter_names
    assert "message_passing.constraint" not in parameter_names
    assert "message_passing.initial_effective_weight" not in parameter_names

    assert "message_passing.fixed_weight" in buffer_names
    assert "message_passing.constraint" in buffer_names
    assert "message_passing.initial_effective_weight" in buffer_names
    assert "message_passing.mask" in buffer_names


def test_graphnn_reset_restores_explicit_initial_weights():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
        quiet=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])

    optimizer.zero_grad()
    loss = model(x).pow(2).sum()
    loss.backward()
    optimizer.step()

    model.apply(_reset_if_available)

    assert _edge_weight(
        model,
        source="feature_a",
        target="hidden_pos",
    ) == pytest.approx(0.25)
    assert _edge_weight(
        model,
        source="feature_b",
        target="hidden_neg",
    ) == pytest.approx(-0.50)
    assert _edge_weight(
        model,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)


def test_graphnn_state_dict_roundtrip_preserves_behavior():
    torch.manual_seed(0)

    edgelist = _metadata_edgelist()

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="graphnn",
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
        backend="graphnn",
        quiet=True,
    )
    reloaded_model.load_state_dict(state_dict)

    assert _edge_constraint_code(
        reloaded_model,
        source="feature_a",
        target="hidden_pos",
    ) == CONSTRAINT_POSITIVE
    assert _edge_constraint_code(
        reloaded_model,
        source="feature_b",
        target="hidden_neg",
    ) == CONSTRAINT_NEGATIVE
    assert _edge_constraint_code(
        reloaded_model,
        source="hidden_pos",
        target="prediction",
    ) == CONSTRAINT_FIXED

    assert _edge_weight(
        reloaded_model,
        source="feature_a",
        target="hidden_pos",
    ) > 0
    assert _edge_weight(
        reloaded_model,
        source="feature_b",
        target="hidden_neg",
    ) < 0
    assert _edge_weight(
        reloaded_model,
        source="hidden_pos",
        target="prediction",
    ) == pytest.approx(0.75)

    torch.testing.assert_close(model(x), reloaded_model(x))