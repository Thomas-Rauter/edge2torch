import pandas as pd
import pytest
import torch

from edge2torch.compile.artifact import CompileArtifact
from edge2torch.compile.execution_plan import FeedforwardExecutionPlan
from edge2torch.compile_graph import compile_graph
from edge2torch.nn.model import EdgeModel
from edge2torch.utils.errors import Edge2TorchError


def test_compile_graph_rejects_feedforward_terminal_outputs_at_multi_depths():
    edgelist = pd.DataFrame(
        {
            "source": ["input", "input", "hidden"],
            "target": ["early_output", "hidden", "late_output"],
        }
    )

    with pytest.raises(
        Edge2TorchError,
        match="terminal output nodes.*same layer depth",
    ):
        compile_graph(
            edgelist=edgelist,
            backend="feedforward",
            quiet=True,
        )


def test_compile_graph_returns_model_and_artifact_for_valid_feedforward_graph():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    assert isinstance(model, EdgeModel)
    assert isinstance(artifact, CompileArtifact)
    assert artifact.backend == "feedforward"
    assert isinstance(artifact.execution_plan, FeedforwardExecutionPlan)


def test_compile_graph_preserves_input_feature_names_in_artifact():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_b", "gene_a"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    _, artifact = compile_graph(edgelist)

    assert artifact.feature_names == ["gene_a", "gene_b"]


def test_compile_graph_builds_expected_layer_structure():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    _, artifact = compile_graph(edgelist)

    assert artifact.node_names_by_layer == {
        "layer_0": ["gene_1", "gene_2"],
        "layer_1": ["pathway_1"],
        "layer_2": ["output_1"],
    }


def test_compile_graph_builds_feedforward_interpretation_metadata():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    _, artifact = compile_graph(edgelist)

    assert artifact.input_nodes == ["gene_1", "gene_2"]
    assert artifact.output_nodes == ["output_1"]
    assert artifact.hidden_nodes == ["pathway_1"]
    assert artifact.interpretation_sites == {
        "layer_1": ["pathway_1"],
        "layer_2": ["output_1"],
    }


def test_compile_graph_expands_feedforward_skip_edges():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "pathway_1", "pathway_2", "gene_1"],
            "target": ["pathway_1", "pathway_2", "output_1", "output_1"],
        }
    )

    _, artifact = compile_graph(edgelist)

    plan = artifact.execution_plan

    pseudo_layer_1 = "__edge2torch_pseudo__gene_1__output_1__layer_1"
    pseudo_layer_2 = "__edge2torch_pseudo__gene_1__output_1__layer_2"

    assert plan.pseudo_nodes == [
        pseudo_layer_1,
        pseudo_layer_2,
    ]

    assert sorted(plan.node_names_by_layer["layer_1"]) == [
        pseudo_layer_1,
        "pathway_1",
    ]

    expanded_edges = plan.expanded_edges.to_dict(orient="records")

    assert {"source": "gene_1", "target": "pathway_1"} in expanded_edges
    assert {"source": "pathway_1", "target": "pathway_2"} in expanded_edges
    assert {"source": "pathway_2", "target": "output_1"} in expanded_edges

    assert {
        "source": "gene_1",
        "target": pseudo_layer_1,
    } in expanded_edges
    assert {
        "source": pseudo_layer_1,
        "target": pseudo_layer_2,
    } in expanded_edges
    assert {
        "source": pseudo_layer_2,
        "target": "output_1",
    } in expanded_edges


def test_compile_graph_returns_model_that_runs_forward_pass():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    x = torch.randn(4, len(artifact.feature_names))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (4, 1)


def test_compile_graph_raises_edge2torcherror_for_non_dataframe_edgelist():
    edgelist = [
        {"source": "gene_1", "target": "pathway_1"},
    ]

    with pytest.raises(Edge2TorchError, match="DataFrame|dataframe"):
        compile_graph(edgelist=edgelist)


def test_compile_graph_raises_edge2torcherror_if_required_columns_are_missing():
    edgelist = pd.DataFrame(
        {
            "src": ["gene_1"],
            "dst": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="source|target"):
        compile_graph(edgelist=edgelist)


def test_compile_graph_raises_edge2torcherror_for_unknown_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="backend|Unsupported backend"):
        compile_graph(edgelist=edgelist, backend="cnn")


def test_compile_backend_rejects_unsupported_backend_directly():
    from edge2torch.compile.compiler import compile_backend
    from edge2torch.graph.io import edgelist_to_graph

    graph = edgelist_to_graph(
        pd.DataFrame(
            {
                "source": ["gene_1"],
                "target": ["pathway_1"],
            }
        )
    )

    with pytest.raises(Edge2TorchError, match="Unsupported backend"):
        compile_backend(graph=graph, backend="cnn")  # type: ignore[arg-type]


def test_compile_graph_raises_edge2torcherror_for_non_bool_quiet():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    with pytest.raises(Edge2TorchError, match="quiet|bool"):
        compile_graph(edgelist=edgelist, quiet="no")


def test_compile_graph_raises_edge2torcherror_for_feedforward_cycle():
    edgelist = pd.DataFrame(
        {
            "source": ["node_a", "node_b"],
            "target": ["node_b", "node_a"],
        }
    )

    with pytest.raises(Edge2TorchError, match="cycles|layered|input node"):
        compile_graph(edgelist=edgelist, backend="feedforward")


def test_compile_graph_raises_edge2torcherror_for_missing_values():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", None],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(Edge2TorchError, match="missing|Missing|source|target"):
        compile_graph(edgelist=edgelist)


def test_compile_graph_raises_edge2torcherror_for_empty_node_names():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "   "],
            "target": ["pathway_1", "pathway_2"],
        }
    )

    with pytest.raises(Edge2TorchError, match="empty|Empty"):
        compile_graph(edgelist=edgelist)
