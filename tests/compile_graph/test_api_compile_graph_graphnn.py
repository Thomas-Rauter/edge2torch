import pandas as pd
import pytest
import torch

from kpnn.compile.artifact import KPNNArtifact
from kpnn.compile.execution_plan import GraphNNExecutionPlan
from kpnn.compile_graph import compile_graph
from kpnn.nn.model import KPNNGraphNNModel
from kpnn.utils.errors import KPNNError


def test_compile_graph_returns_graphnn_model_and_artifact():
    edgelist = pd.DataFrame(
        {
            "source": ["node_a", "node_b"],
            "target": ["node_b", "node_a"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="graphnn")

    assert isinstance(model, KPNNGraphNNModel)
    assert isinstance(artifact, KPNNArtifact)
    assert artifact.backend == "graphnn"
    assert isinstance(artifact.execution_plan, GraphNNExecutionPlan)


def test_compile_graph_graphnn_allows_cycles():
    edgelist = pd.DataFrame(
        {
            "source": ["node_a", "node_b"],
            "target": ["node_b", "node_a"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="graphnn")

    assert model is not None
    assert artifact is not None


def test_compile_graph_graphnn_returns_expected_feature_names():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    _, artifact = compile_graph(edgelist, backend="graphnn")

    assert artifact.feature_names == ["gene_1", "gene_2"]


def test_graphnn_model_runs_forward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="graphnn")

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 1)


def test_graphnn_model_raises_for_wrong_input_width():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="graphnn")

    x = torch.randn(4, len(artifact.feature_names) + 1)

    with pytest.raises(KPNNError, match="wrong number of features"):
        model(x)
