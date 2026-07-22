import pandas as pd
import pytest
import torch
from torch import nn

from edge2torch.compile_graph import compile_graph
from edge2torch.customize_model import customize_model
from edge2torch.utils.errors import Edge2TorchError


def test_customize_model_wraps_feedforward_model_and_runs():
    edgelist = pd.DataFrame(
        {
            "source": [
                "gene_1",
                "pathway_1",
                "pathway_2",
                "gene_1",
            ],
            "target": [
                "pathway_1",
                "pathway_2",
                "output_1",
                "output_1",
            ],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="feedforward",
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        dropout=0.1,
        head=nn.Linear(1, 2),
    )

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 2)


def test_customize_model_wraps_state_update_model_and_runs():
    edgelist = pd.DataFrame(
        {
            "source": [
                "gene_1",
                "node_a",
                "node_b",
                "node_b",
            ],
            "target": [
                "node_a",
                "node_b",
                "node_a",
                "output_1",
            ],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="state_update",
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        dropout=0.1,
        head=nn.Linear(1, 2),
    )

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 2)


def test_customized_feedforward_model_supports_backward_pass():
    edgelist = pd.DataFrame(
        {
            "source": [
                "gene_1",
                "pathway_1",
                "pathway_2",
                "gene_1",
            ],
            "target": [
                "pathway_1",
                "pathway_2",
                "output_1",
                "output_1",
            ],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="feedforward",
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        head=nn.Linear(1, 1),
    )

    x = torch.randn(5, len(artifact.feature_names))
    y = model(x)
    loss = y.mean()

    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)


def test_customized_state_update_model_supports_backward_pass():
    edgelist = pd.DataFrame(
        {
            "source": [
                "gene_1",
                "node_a",
                "node_b",
                "node_b",
            ],
            "target": [
                "node_a",
                "node_b",
                "node_a",
                "output_1",
            ],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="state_update",
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        head=nn.Linear(1, 1),
    )

    x = torch.randn(5, len(artifact.feature_names))
    y = model(x)
    loss = y.mean()

    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)


def test_customized_state_update_model_supports_optimizer_step():
    edgelist = pd.DataFrame(
        {
            "source": [
                "gene_1",
                "gene_2",
                "node_a",
                "node_b",
            ],
            "target": [
                "node_a",
                "node_a",
                "node_b",
                "output_1",
            ],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="state_update",
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        head=nn.Linear(1, 1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(5, len(artifact.feature_names))
    y = model(x)
    loss = y.sum()

    before = [param.detach().clone() for param in model.parameters()]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = list(model.parameters())

    assert any(
        not torch.equal(param_before, param_after)
        for param_before, param_after in zip(before, after)
    )


def test_customize_model_accepts_no_optional_components():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["output_1"],
        }
    )

    base_model, artifact = compile_graph(
        edgelist,
        backend="feedforward",
    )

    model = customize_model(model=base_model)

    x = torch.randn(3, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)


def test_customize_model_rejects_invalid_activation():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["output_1"],
        }
    )

    base_model, _ = compile_graph(
        edgelist,
        backend="feedforward",
    )

    with pytest.raises(Edge2TorchError, match="activation"):
        customize_model(
            model=base_model,
            activation="relu",
        )


def test_customize_model_rejects_invalid_dropout_type():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["output_1"],
        }
    )

    base_model, _ = compile_graph(
        edgelist,
        backend="feedforward",
    )

    with pytest.raises(Edge2TorchError, match="dropout"):
        customize_model(
            model=base_model,
            dropout="0.2",
        )


def test_customize_model_rejects_invalid_dropout_range():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["output_1"],
        }
    )

    base_model, _ = compile_graph(
        edgelist,
        backend="feedforward",
    )

    with pytest.raises(Edge2TorchError, match="0 <= dropout < 1"):
        customize_model(
            model=base_model,
            dropout=1.0,
        )


def test_customize_model_rejects_invalid_head():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["output_1"],
        }
    )

    base_model, _ = compile_graph(
        edgelist,
        backend="feedforward",
    )

    with pytest.raises(Edge2TorchError, match="head"):
        customize_model(
            model=base_model,
            head="linear",
        )


def test_customize_model_rejects_mismatched_head_width():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    base_model, _ = compile_graph(
        edgelist,
        backend="feedforward",
        quiet=True,
    )

    with pytest.raises(
        Edge2TorchError,
        match="in_features match the model output width",
    ):
        customize_model(
            model=base_model,
            head=nn.Linear(99, 1),
        )
