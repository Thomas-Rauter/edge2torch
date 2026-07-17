import pandas as pd
import pytest
import torch

from edge2torch import compile_graph
from edge2torch.nn.masked_linear import ConstrainedMaskedLinear
from edge2torch.utils.errors import Edge2TorchError


def _edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["feature_a", "feature_b", "hidden"],
            "target": ["hidden", "hidden", "prediction"],
        }
    )


def _metadata_edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["feature_a", "feature_b", "hidden"],
            "target": ["hidden", "hidden", "prediction"],
            "initial_weight": [0.25, -0.50, 0.75],
            "constraint": ["positive", "negative", "fixed"],
        }
    )


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_compile_graph_bias_true_keeps_bias_parameters(backend: str):
    model, _ = compile_graph(
        edgelist=_edgelist(),
        backend=backend,
        quiet=True,
        bias=True,
    )

    if backend == "feedforward":
        for block in model.blocks:
            assert block.linear.bias is not None
    elif backend == "state_update":
        assert model.state_linear.bias is not None
    else:
        assert model.state_linear.bias is not None


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_compile_graph_bias_false_removes_bias_parameters(backend: str):
    model, _ = compile_graph(
        edgelist=_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
    )

    if backend == "feedforward":
        for block in model.blocks:
            assert block.linear.bias is None
    elif backend == "state_update":
        assert model.state_linear.bias is None
    else:
        assert model.state_linear.bias is None


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_compile_graph_bias_false_excludes_bias_from_parameters(
    backend: str,
):
    model, _ = compile_graph(
        edgelist=_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
    )

    parameter_names = {name for name, _ in model.named_parameters()}

    assert all(not name.endswith(".bias") for name in parameter_names)


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_compile_graph_bias_true_includes_bias_in_parameters(backend: str):
    model, _ = compile_graph(
        edgelist=_edgelist(),
        backend=backend,
        quiet=True,
        bias=True,
    )

    parameter_names = {name for name, _ in model.named_parameters()}

    assert any(name.endswith(".bias") for name in parameter_names)


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_bias_false_works_with_edge_metadata(backend: str):
    model, _ = compile_graph(
        edgelist=_metadata_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
    )

    if backend == "feedforward":
        for block in model.blocks:
            assert isinstance(block.linear, ConstrainedMaskedLinear)
            assert block.linear.bias is None
    elif backend == "state_update":
        assert isinstance(model.state_linear, ConstrainedMaskedLinear)
        assert model.state_linear.bias is None
    else:
        assert isinstance(model.state_linear, ConstrainedMaskedLinear)
        assert model.state_linear.bias is None


@pytest.mark.parametrize(
    "backend",
    ["feedforward", "state_update"],
)
def test_bias_false_forward_pass_runs(backend: str):
    model, artifact = compile_graph(
        edgelist=_edgelist(),
        backend=backend,
        quiet=True,
        bias=False,
    )

    x = torch.ones(2, len(artifact.feature_names))
    output = model(x)

    assert output.shape == (2, 1)


def test_compile_graph_rejects_non_boolean_bias():
    with pytest.raises(Edge2TorchError, match="'bias' must be a boolean"):
        compile_graph(
            edgelist=_edgelist(),
            backend="feedforward",
            quiet=True,
            bias="false",
        )
