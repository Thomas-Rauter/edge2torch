import pandas as pd
import pytest
import torch

from edge2torch.compile.artifact import CompileArtifact
from edge2torch.compile.execution_plan import StateUpdateExecutionPlan
from edge2torch.compile_graph import compile_graph
from edge2torch.nn.model import StateUpdateEdgeModel
from edge2torch.utils.errors import Edge2TorchError


def test_compile_graph_returns_state_update_model_and_artifact():
    edgelist = pd.DataFrame(
        {
            "source": ["input_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update")

    assert isinstance(model, StateUpdateEdgeModel)
    assert isinstance(artifact, CompileArtifact)
    assert artifact.backend == "state_update"
    assert isinstance(artifact.execution_plan, StateUpdateExecutionPlan)


def test_compile_graph_state_update_allows_cycles():
    edgelist = pd.DataFrame(
        {
            "source": ["input_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update")

    assert model is not None
    assert artifact is not None


def test_compile_graph_state_update_returns_expected_feature_names():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    _, artifact = compile_graph(edgelist, backend="state_update")

    assert artifact.feature_names == ["gene_1", "gene_2"]


def test_state_update_model_runs_forward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update")

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 1)


def test_state_update_model_raises_for_wrong_input_width():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update")

    x = torch.randn(4, len(artifact.feature_names) + 1)

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        model(x)


def test_compile_graph_state_update_rejects_unreachable_output_relevant_nodes():
    edgelist = pd.DataFrame(
        {
            "source": ["feature", "a", "b", "b"],
            "target": ["prediction", "b", "a", "prediction"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="Unreachable output-relevant node",
    ):
        compile_graph(
            edgelist=edgelist,
            backend="state_update",
            quiet=True,
        )
