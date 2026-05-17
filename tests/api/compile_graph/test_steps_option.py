import pandas as pd
import pytest
import torch

from edge2torch import compile_graph
from edge2torch.utils.errors import Edge2TorchError


def _state_update_edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["feature_a", "hidden_1", "hidden_2"],
            "target": ["hidden_1", "hidden_2", "prediction"],
        }
    )


@pytest.mark.parametrize("backend", ["recurrent", "graphnn"])
def test_compile_graph_sets_steps_for_state_update_backends(
    backend: str,
):
    model, _ = compile_graph(
        edgelist=_state_update_edgelist(),
        backend=backend,
        quiet=True,
        steps=5,
    )

    assert model.steps == 5


@pytest.mark.parametrize("backend", ["recurrent", "graphnn"])
def test_compile_graph_default_steps_is_three(
    backend: str,
):
    model, _ = compile_graph(
        edgelist=_state_update_edgelist(),
        backend=backend,
        quiet=True,
    )

    assert model.steps == 3


@pytest.mark.parametrize("backend", ["recurrent", "graphnn"])
def test_steps_changes_state_update_forward_behavior(
    backend: str,
):
    torch.manual_seed(0)

    model_one_step, artifact = compile_graph(
        edgelist=_state_update_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
        steps=1,
    )
    model_three_steps, _ = compile_graph(
        edgelist=_state_update_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
        steps=3,
    )

    model_three_steps.load_state_dict(model_one_step.state_dict())

    x = torch.ones(2, len(artifact.feature_names))

    output_one_step = model_one_step(x)
    output_three_steps = model_three_steps(x)

    assert output_one_step.shape == (2, 1)
    assert output_three_steps.shape == (2, 1)
    assert not torch.allclose(output_one_step, output_three_steps)


def test_compile_graph_allows_default_steps_for_feedforward():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "hidden"],
            "target": ["hidden", "prediction"],
        }
    )

    model, _ = compile_graph(
        edgelist=edgelist,
        backend="feedforward",
        quiet=True,
        steps=3,
    )

    assert model.backend == "feedforward"


def test_compile_graph_rejects_non_default_steps_for_feedforward():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "hidden"],
            "target": ["hidden", "prediction"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="'steps' is only used by the 'recurrent' and 'graphnn'",
    ):
        compile_graph(
            edgelist=edgelist,
            backend="feedforward",
            quiet=True,
            steps=5,
        )


@pytest.mark.parametrize("steps", [0, -1])
def test_compile_graph_rejects_non_positive_steps(steps: int):
    with pytest.raises(
        Edge2TorchError,
        match="'steps' must be a positive integer",
    ):
        compile_graph(
            edgelist=_state_update_edgelist(),
            backend="recurrent",
            quiet=True,
            steps=steps,
        )


@pytest.mark.parametrize("steps", [1.5, "3", None])
def test_compile_graph_rejects_non_integer_steps(steps):
    with pytest.raises(Edge2TorchError, match="'steps' must be an integer"):
        compile_graph(
            edgelist=_state_update_edgelist(),
            backend="recurrent",
            quiet=True,
            steps=steps,
        )
