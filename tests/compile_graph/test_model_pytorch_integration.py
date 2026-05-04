import pandas as pd
import torch
from torch import nn

from edge2torch.compile_graph import compile_graph


def test_compiled_model_runs_forward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="feedforward")

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 1)


def test_compiled_model_supports_backward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="feedforward")

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)
    loss = y.sum()

    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)


def test_compiled_model_supports_optimizer_step():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="feedforward")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(4, len(artifact.feature_names))
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


def test_compiled_model_can_be_wrapped_in_plain_pytorch_module():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    base_model, artifact = compile_graph(edgelist, backend="feedforward")

    class WrappedModel(nn.Module):
        def __init__(self, compiled_model):
            super().__init__()
            self.compiled_model = compiled_model
            self.activation = nn.ReLU()
            self.head = nn.Linear(1, 2)

        def forward(self, x):
            x = self.compiled_model(x)
            x = self.activation(x)
            x = self.head(x)
            return x

    model = WrappedModel(base_model)

    x = torch.randn(5, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (5, 2)


def test_wrapped_compiled_model_supports_backward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    base_model, artifact = compile_graph(edgelist, backend="feedforward")

    class WrappedModel(nn.Module):
        def __init__(self, compiled_model):
            super().__init__()
            self.compiled_model = compiled_model
            self.activation = nn.ReLU()
            self.head = nn.Linear(1, 1)

        def forward(self, x):
            x = self.compiled_model(x)
            x = self.activation(x)
            x = self.head(x)
            return x

    model = WrappedModel(base_model)

    x = torch.randn(6, len(artifact.feature_names))
    y = model(x)
    loss = y.mean()

    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)


def test_compiled_model_runs_with_skip_edge_expansion():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "pathway_1", "pathway_2", "gene_1"],
            "target": ["pathway_1", "pathway_2", "output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="feedforward")

    x = torch.randn(3, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
