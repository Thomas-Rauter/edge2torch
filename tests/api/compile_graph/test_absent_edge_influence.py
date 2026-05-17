import pandas as pd
import pytest
import torch

from edge2torch import compile_graph


def _independent_paths_edgelist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [
                "feature_a",
                "hidden_a",
                "feature_b",
                "hidden_b",
            ],
            "target": [
                "hidden_a",
                "prediction",
                "hidden_b",
                "decoy",
            ],
            "initial_weight": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "constraint": [
                "fixed",
                "fixed",
                "fixed",
                "fixed",
            ],
        }
    )


def _compile_model(backend: str):
    kwargs = {
        "edgelist": _independent_paths_edgelist(),
        "backend": backend,
        "quiet": True,
        "bias": False,
    }

    if backend in {"recurrent", "graphnn"}:
        kwargs["steps"] = 2

    return compile_graph(**kwargs)


def _output_index(model: torch.nn.Module, node_name: str) -> int:
    if model.backend == "feedforward":
        final_block = model.blocks[-1]
        return final_block.output_node_names.index(node_name)

    return model.output_node_names.index(node_name)


@pytest.mark.parametrize("backend", ["feedforward", "recurrent", "graphnn"])
def test_absent_edge_has_zero_forward_influence(backend: str):
    model, artifact = _compile_model(backend)

    prediction_idx = _output_index(model, "prediction")
    decoy_idx = _output_index(model, "decoy")

    feature_a_idx = artifact.feature_names.index("feature_a")
    feature_b_idx = artifact.feature_names.index("feature_b")

    x_low = torch.zeros(1, len(artifact.feature_names))
    x_high = torch.zeros(1, len(artifact.feature_names))

    x_low[0, feature_a_idx] = 2.0
    x_low[0, feature_b_idx] = 0.0

    x_high[0, feature_a_idx] = 2.0
    x_high[0, feature_b_idx] = 100.0

    output_low = model(x_low)
    output_high = model(x_high)

    torch.testing.assert_close(
        output_low[:, prediction_idx],
        output_high[:, prediction_idx],
    )

    assert not torch.allclose(
        output_low[:, decoy_idx],
        output_high[:, decoy_idx],
    )


@pytest.mark.parametrize("backend", ["feedforward", "recurrent", "graphnn"])
def test_absent_edge_has_zero_input_gradient(backend: str):
    model, artifact = _compile_model(backend)

    prediction_idx = _output_index(model, "prediction")

    feature_a_idx = artifact.feature_names.index("feature_a")
    feature_b_idx = artifact.feature_names.index("feature_b")

    x = torch.zeros(1, len(artifact.feature_names), requires_grad=True)
    x.data[0, feature_a_idx] = 2.0
    x.data[0, feature_b_idx] = 100.0

    prediction = model(x)[:, prediction_idx].sum()
    prediction.backward()

    assert x.grad is not None
    assert x.grad[0, feature_a_idx] != 0
    assert x.grad[0, feature_b_idx] == pytest.approx(0.0)
