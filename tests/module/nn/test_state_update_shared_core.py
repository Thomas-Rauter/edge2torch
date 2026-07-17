"""Tests for the public state_update backend."""

from pathlib import Path

import pandas as pd
import torch

from edge2torch.compile_graph import compile_graph
from edge2torch.nn.model import StateUpdateEdgeModel

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "edgelists"


def test_state_update_compile_uses_state_linear_module():
    edgelist = pd.DataFrame(
        {
            "source": ["input_1", "hidden_1"],
            "target": ["hidden_1", "output_1"],
        }
    )

    model, artifact = compile_graph(
        edgelist,
        backend="state_update",
        quiet=True,
    )

    assert isinstance(model, StateUpdateEdgeModel)
    assert model.backend == "state_update"
    assert artifact.backend == "state_update"
    assert "state_linear.weight" in dict(model.named_parameters())


def test_state_update_interpretation_sites_use_step_prefix():
    edgelist = pd.read_csv(FIXTURES / "state_update_cycle.csv")

    model, artifact = compile_graph(
        edgelist,
        backend="state_update",
        steps=3,
        quiet=True,
    )

    assert model._edge2torch_list_interpretation_site_ids() == [
        "step_1",
        "step_2",
        "step_3",
    ]
    assert list(artifact.interpretation_sites.keys()) == [
        "step_1",
        "step_2",
        "step_3",
    ]


def test_state_update_cycle_fixture_compiles_and_runs_forward():
    edgelist = pd.read_csv(FIXTURES / "state_update_cycle.csv")

    model, artifact = compile_graph(
        edgelist,
        backend="state_update",
        steps=3,
        quiet=True,
    )

    x = torch.randn(2, len(artifact.feature_names))
    y = model(x)

    assert y.shape == (2, len(artifact.execution_plan.output_node_names))
