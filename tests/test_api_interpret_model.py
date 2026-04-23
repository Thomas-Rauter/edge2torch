import pandas as pd
import pytest
import torch

from kpnn.compile_graph import compile_graph
from kpnn.interpret_model import interpret_model
from kpnn.utils.errors import KPNNError


def test_interpret_model_returns_feature_dataframe():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2, 0.3],
            "gene_2": [1.0, 1.1, 1.2],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="integrated_gradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.index) == ["cell_1", "cell_2", "cell_3"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_returns_node_dataframes_for_layer_conductance():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        },
        index=["cell_1", "cell_2"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="layer_conductance",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"layer_1", "layer_2"}

    assert isinstance(result["layer_1"], pd.DataFrame)
    assert isinstance(result["layer_2"], pd.DataFrame)

    assert result["layer_1"].shape == (2, 1)
    assert result["layer_2"].shape == (2, 1)

    assert list(result["layer_1"].index) == ["cell_1", "cell_2"]
    assert list(result["layer_2"].index) == ["cell_1", "cell_2"]

    assert list(result["layer_1"].columns) == ["pathway_1"]
    assert list(result["layer_2"].columns) == ["output_1"]


def test_interpret_model_returns_node_dataframes_for_layer_ig():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        },
        index=["cell_1", "cell_2"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="layer_integrated_gradients",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"layer_1", "layer_2"}

    assert result["layer_1"].shape == (2, 1)
    assert result["layer_2"].shape == (2, 1)

    assert list(result["layer_1"].columns) == ["pathway_1"]
    assert list(result["layer_2"].columns) == ["output_1"]


def test_interpret_model_accepts_tensor_input_for_feature_target():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = torch.tensor(
        [[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]],
        dtype=torch.float32,
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="integrated_gradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_reorders_dataframe_columns_to_feature_names():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_b": [1.0, 1.1],
            "gene_a": [0.1, 0.2],
        },
        index=["cell_1", "cell_2"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="integrated_gradients",
    )

    assert list(result.columns) == artifact.feature_names


def test_interpret_model_raises_for_incompatible_feature_method():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame({"gene_1": [1.0]})

    with pytest.raises(KPNNError, match="compatible"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="layer_conductance",
        )


def test_interpret_model_raises_for_incompatible_node_method():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1"],
            "target": ["pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame({"gene_1": [1.0]})

    with pytest.raises(KPNNError, match="compatible"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="nodes",
            method="integrated_gradients",
        )


def test_interpret_model_raises_for_missing_dataframe_features():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
        }
    )

    with pytest.raises(KPNNError, match="missing required feature"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="integrated_gradients",
        )


def test_interpret_model_raises_for_wrong_tensor_feature_count():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = torch.randn(3, 3)

    with pytest.raises(KPNNError, match="wrong number of features"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="integrated_gradients",
        )


def test_interpret_model_hides_pseudo_nodes_in_node_results():
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

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
        },
        index=["cell_1", "cell_2"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="layer_conductance",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"layer_1", "layer_2", "layer_3"}

    assert list(result["layer_1"].columns) == ["pathway_1"]
    assert list(result["layer_2"].columns) == ["pathway_2"]
    assert list(result["layer_3"].columns) == ["output_1"]

    assert "pseudo__gene_1__output_1__layer_1" not in result["layer_1"].columns
    assert "pseudo__gene_1__output_1__layer_2" not in result["layer_2"].columns

    assert result["layer_1"].shape == (2, 1)
    assert result["layer_2"].shape == (2, 1)
    assert result["layer_3"].shape == (2, 1)


def test_interpret_model_hides_pseudo_nodes_for_layer_ig():
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

    model, artifact = compile_graph(edgelist)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
        },
        index=["cell_1", "cell_2"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="layer_integrated_gradients",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"layer_1", "layer_2", "layer_3"}

    assert list(result["layer_1"].columns) == ["pathway_1"]
    assert list(result["layer_2"].columns) == ["pathway_2"]
    assert list(result["layer_3"].columns) == ["output_1"]

    assert "pseudo__gene_1__output_1__layer_1" not in result["layer_1"].columns
    assert "pseudo__gene_1__output_1__layer_2" not in result["layer_2"].columns

    assert result["layer_1"].shape == (2, 1)
    assert result["layer_2"].shape == (2, 1)
    assert result["layer_3"].shape == (2, 1)
