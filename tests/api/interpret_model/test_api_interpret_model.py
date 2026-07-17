import numpy as np
import pandas as pd
import pytest
import torch

try:
    import anndata as ad
except ImportError:
    ad = None

from edge2torch.compile_graph import compile_graph
from edge2torch.interpret_model import interpret_model
from edge2torch.utils.errors import Edge2TorchError


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
        method="IntegratedGradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.index) == ["cell_1", "cell_2", "cell_3"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_returns_node_summary_for_layer_conductance():
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
        method="LayerConductance",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["pathway_1"]


def test_interpret_model_returns_node_site_tables_for_layer_conductance():
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
        method="LayerConductance",
        level="sites",
        nodes="non_input",
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


def test_interpret_model_returns_node_summary_for_layer_ig():
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
        method="LayerIntegratedGradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.columns) == ["pathway_1"]


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
        method="IntegratedGradients",
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
        method="IntegratedGradients",
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

    with pytest.raises(Edge2TorchError, match="compatible"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="LayerConductance",
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

    with pytest.raises(Edge2TorchError, match="compatible"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="nodes",
            method="IntegratedGradients",
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

    with pytest.raises(Edge2TorchError, match="missing required feature"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
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

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
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
        method="LayerConductance",
        level="sites",
        nodes="non_input",
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
        method="LayerIntegratedGradients",
        level="sites",
        nodes="non_input",
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


@pytest.mark.skipif(ad is None, reason="anndata not installed")
def test_interpret_model_accepts_anndata_for_feature_target():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = ad.AnnData(
        X=np.array([[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]], dtype=float),
    )
    data.var_names = ["gene_1", "gene_2"]
    data.obs_names = ["cell_1", "cell_2", "cell_3"]

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="IntegratedGradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.index) == ["cell_1", "cell_2", "cell_3"]
    assert list(result.columns) == ["gene_1", "gene_2"]


@pytest.mark.skipif(ad is None, reason="anndata not installed")
def test_interpret_model_reorders_anndata_var_names():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = ad.AnnData(
        X=np.array([[1.0, 0.1], [1.1, 0.2]], dtype=float),
    )
    data.var_names = ["gene_b", "gene_a"]
    data.obs_names = ["cell_1", "cell_2"]

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="IntegratedGradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == artifact.feature_names


@pytest.mark.skipif(ad is None, reason="anndata not installed")
def test_interpret_model_raises_for_missing_anndata_features():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist)

    data = ad.AnnData(
        X=np.array([[0.1], [0.2]], dtype=float),
    )
    data.var_names = ["gene_1"]
    data.obs_names = ["cell_1", "cell_2"]

    with pytest.raises(Edge2TorchError, match="missing required variable name"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
        )


def test_interpret_model_supports_feature_target_for_state_update_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update")

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2, 0.3],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="IntegratedGradients",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 1)
    assert list(result.index) == ["cell_1", "cell_2", "cell_3"]
    assert list(result.columns) == ["gene_1"]


def test_interpret_model_supports_node_target_for_state_update_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update", steps=2)

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
        method="LayerConductance",
        quiet=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["node_a", "node_b"]
    assert result.shape == (2, 2)


def test_interpret_model_supports_node_sites_for_state_update_backend():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, backend="state_update", steps=2)

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
        method="LayerConductance",
        quiet=True,
        level="sites",
        nodes="non_input",
    )

    assert isinstance(result, dict)
    assert list(result.keys()) == ["step_1", "step_2"]
    assert list(result["step_1"].columns) == ["node_a", "node_b", "output_1"]
    assert result["step_1"].shape == (2, 3)


def test_interpret_model_accepts_feature_constructor_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        target="features",
        method="IntegratedGradients",
        quiet=True,
        constructor_kwargs={"multiply_by_inputs": True},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_accepts_feature_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        target="features",
        method="IntegratedGradients",
        quiet=True,
        attribute_kwargs={"n_steps": 4},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_accepts_node_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        method="LayerIntegratedGradients",
        quiet=True,
        attribute_kwargs={"n_steps": 4},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["pathway_1"]


def test_interpret_model_rejects_invalid_constructor_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(Edge2TorchError, match="constructor_kwargs"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=["not", "a", "dict"],
        )


def test_interpret_model_rejects_invalid_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(Edge2TorchError, match="attribute_kwargs"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
            quiet=True,
            attribute_kwargs=["not", "a", "dict"],
        )


def test_interpret_model_rejects_return_convergence_delta():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(Edge2TorchError, match="return_convergence_delta=True"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="IntegratedGradients",
            quiet=True,
            attribute_kwargs={"return_convergence_delta": True},
        )


@pytest.mark.filterwarnings(
    "ignore:Input Tensor 0 did not already require gradients.*:UserWarning"
)
@pytest.mark.parametrize(
    "method",
    [
        "IntegratedGradients",
        "Saliency",
        "InputXGradient",
        "FeatureAblation",
        "FeaturePermutation",
        "ShapleyValueSampling",
        "ShapleyValues",
    ],
)
def test_interpret_model_supports_basic_feature_methods(method):
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        target="features",
        method=method,
        quiet=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_supports_gradient_shap_with_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        },
        index=["cell_1", "cell_2"],
    )

    baselines = torch.zeros(2, 2)

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="GradientShap",
        quiet=True,
        attribute_kwargs={
            "baselines": baselines,
            "n_samples": 2,
            "stdevs": 0.0,
        },
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_supports_occlusion_with_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        target="features",
        method="Occlusion",
        quiet=True,
        attribute_kwargs={
            "sliding_window_shapes": (1,),
        },
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_rejects_constructor_kwargs_for_saliency():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(
        Edge2TorchError, match="does not support constructor_kwargs"
    ):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="Saliency",
            quiet=True,
            constructor_kwargs={"multiply_by_inputs": True},
        )


def test_interpret_model_accepts_constructor_kwargs_for_integrated_gradients():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        target="features",
        method="IntegratedGradients",
        quiet=True,
        constructor_kwargs={"multiply_by_inputs": True},
        attribute_kwargs={"n_steps": 4},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["gene_1", "gene_2"]


def test_interpret_model_rejects_unknown_feature_method():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(Edge2TorchError, match="Unsupported method"):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="features",
            method="not_a_captum_method",
            quiet=True,
        )


@pytest.mark.parametrize(
    "method",
    [
        "LayerConductance",
        "LayerActivation",
        "LayerGradientXActivation",
        "LayerFeatureAblation",
        "LayerFeaturePermutation",
        "LayerIntegratedGradients",
    ],
)
def test_interpret_model_supports_feedforward_node_methods(method):
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        method=method,
        quiet=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["pathway_1"]


def test_interpret_model_supports_layer_gradient_shap_with_attribute_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        },
        index=["cell_1", "cell_2"],
    )

    baselines = torch.zeros(2, 2)

    result = interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="nodes",
        method="LayerGradientShap",
        quiet=True,
        attribute_kwargs={
            "baselines": baselines,
            "n_samples": 2,
            "stdevs": 0.0,
        },
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["pathway_1"]


def test_interpret_model_accepts_layer_ig_constructor_kwargs():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2", "pathway_1"],
            "target": ["pathway_1", "pathway_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

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
        method="LayerIntegratedGradients",
        quiet=True,
        constructor_kwargs={"multiply_by_inputs": True},
        attribute_kwargs={"n_steps": 4},
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert list(result.index) == ["cell_1", "cell_2"]
    assert list(result.columns) == ["pathway_1"]


def test_interpret_model_rejects_constructor_kwargs_for_layer_activation():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            "gene_1": [0.1, 0.2],
            "gene_2": [1.0, 1.1],
        }
    )

    with pytest.raises(
        Edge2TorchError, match="does not support constructor_kwargs"
    ):
        interpret_model(
            model=model,
            artifact=artifact,
            data=data,
            target="nodes",
            method="LayerActivation",
            quiet=True,
            constructor_kwargs={"multiply_by_inputs": True},
        )


def test_interpret_model_restores_training_mode_after_interpretation():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)
    model.train()

    data = pd.DataFrame(
        {
            "gene_1": [1.0, 2.0],
            "gene_2": [3.0, 4.0],
        }
    )

    interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="IntegratedGradients",
        quiet=True,
        attribute_kwargs={"n_steps": 8},
    )

    assert model.training is True


def test_interpret_model_restores_eval_mode_after_interpretation():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["output_1", "output_1"],
        }
    )

    model, artifact = compile_graph(edgelist, quiet=True)
    model.eval()

    data = pd.DataFrame(
        {
            "gene_1": [1.0, 2.0],
            "gene_2": [3.0, 4.0],
        }
    )

    interpret_model(
        model=model,
        artifact=artifact,
        data=data,
        target="features",
        method="IntegratedGradients",
        quiet=True,
        attribute_kwargs={"n_steps": 8},
    )

    assert model.training is False
