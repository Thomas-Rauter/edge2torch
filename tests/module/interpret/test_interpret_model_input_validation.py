import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from edge2torch.interpret.input_validation import (
    _validate_interpret_anndata,
    _validate_interpret_artifact,
    _validate_interpret_data,
    _validate_interpret_dataframe,
    _validate_interpret_model,
    _validate_interpret_options,
    _validate_interpret_tensor,
)
from edge2torch.utils.errors import Edge2TorchError


class _Artifact:
    def __init__(
        self,
        *,
        backend="feedforward",
        feature_names=None,
        node_names_by_layer=None,
        interpretation_sites=None,
        input_nodes=None,
        output_nodes=None,
        hidden_nodes=None,
        execution_plan=object(),
    ):
        self.backend = backend
        self.feature_names = (
            ["gene_a", "gene_b"] if feature_names is None else feature_names
        )
        self.node_names_by_layer = (
            {
                "layer_0": ["gene_a", "gene_b"],
                "layer_1": ["output_1"],
            }
            if node_names_by_layer is None
            else node_names_by_layer
        )
        self.interpretation_sites = (
            {"layer_1": ["output_1"]}
            if interpretation_sites is None
            else interpretation_sites
        )
        self.input_nodes = (
            ["gene_a", "gene_b"] if input_nodes is None else input_nodes
        )
        self.output_nodes = (
            ["output_1"] if output_nodes is None else output_nodes
        )
        self.hidden_nodes = [] if hidden_nodes is None else hidden_nodes
        self.execution_plan = execution_plan


class _ModuleWithoutVisibleForward(nn.Module):
    def __getattribute__(self, name):
        if name == "forward":
            raise AttributeError
        return super().__getattribute__(name)


# _validate_interpret_options --------------------------------------------------


def test_validate_interpret_options_rejects_non_bool_quiet():
    with pytest.raises(Edge2TorchError, match="'quiet' must be a boolean"):
        _validate_interpret_options(
            target="features",
            method="IntegratedGradients",
            quiet="yes",
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_non_string_target():
    with pytest.raises(Edge2TorchError, match="'target' must be a string"):
        _validate_interpret_options(
            target=1,
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_unknown_target():
    with pytest.raises(Edge2TorchError, match="Unsupported target"):
        _validate_interpret_options(
            target="edges",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_non_string_method():
    with pytest.raises(Edge2TorchError, match="'method' must be a string"):
        _validate_interpret_options(
            target="features",
            method=1,
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_unknown_method():
    with pytest.raises(Edge2TorchError, match="Unsupported method"):
        _validate_interpret_options(
            target="features",
            method="not_a_method",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_node_method_for_feature_target():
    with pytest.raises(
        Edge2TorchError,
        match="not compatible with target='features'",
    ):
        _validate_interpret_options(
            target="features",
            method="LayerConductance",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_feature_method_for_node_target():
    with pytest.raises(
        Edge2TorchError,
        match="not compatible with target='nodes'",
    ):
        _validate_interpret_options(
            target="nodes",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_non_dict_constructor_kwargs():
    with pytest.raises(
        Edge2TorchError,
        match="'constructor_kwargs' must be a dictionary or None",
    ):
        _validate_interpret_options(
            target="features",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=["bad"],
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_feature_constructor_kwargs():
    with pytest.raises(
        Edge2TorchError,
        match="does not support constructor_kwargs",
    ):
        _validate_interpret_options(
            target="features",
            method="Saliency",
            quiet=True,
            constructor_kwargs={"bad": True},
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_node_constructor_kwargs():
    with pytest.raises(
        Edge2TorchError,
        match="does not support constructor_kwargs",
    ):
        _validate_interpret_options(
            target="nodes",
            method="LayerActivation",
            quiet=True,
            constructor_kwargs={"bad": True},
            attribute_kwargs=None,
        )


def test_validate_interpret_options_rejects_non_dict_attribute_kwargs():
    with pytest.raises(
        Edge2TorchError,
        match="'attribute_kwargs' must be a dictionary or None",
    ):
        _validate_interpret_options(
            target="features",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=["bad"],
        )


def test_validate_interpret_options_rejects_return_convergence_delta():
    with pytest.raises(
        Edge2TorchError,
        match="return_convergence_delta=True",
    ):
        _validate_interpret_options(
            target="features",
            method="IntegratedGradients",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs={"return_convergence_delta": True},
        )


def test_validate_interpret_options_rejects_unknown_level():
    with pytest.raises(Edge2TorchError, match="Unsupported level"):
        _validate_interpret_options(
            target="nodes",
            method="LayerConductance",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
            level="per_layer",
        )


def test_validate_interpret_options_rejects_unknown_nodes_filter():
    with pytest.raises(Edge2TorchError, match="Unsupported nodes filter"):
        _validate_interpret_options(
            target="nodes",
            method="LayerConductance",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
            nodes="outputs_only",
        )


def test_validate_interpret_options_rejects_unknown_site_aggregation():
    with pytest.raises(Edge2TorchError, match="Unsupported site_aggregation"):
        _validate_interpret_options(
            target="nodes",
            method="LayerConductance",
            quiet=True,
            constructor_kwargs=None,
            attribute_kwargs=None,
            site_aggregation="sum",
        )


# _validate_interpret_model ----------------------------------------------------


def test_validate_interpret_model_rejects_non_module_model():
    with pytest.raises(Edge2TorchError, match="torch.nn.Module"):
        _validate_interpret_model(object())


def test_validate_interpret_model_rejects_module_without_visible_forward():
    model = _ModuleWithoutVisibleForward()

    with pytest.raises(Edge2TorchError, match="forward method"):
        _validate_interpret_model(model)


# _validate_interpret_artifact -------------------------------------------------


def test_validate_interpret_artifact_rejects_missing_required_attrs():
    with pytest.raises(
        Edge2TorchError,
        match="missing required attribute",
    ):
        _validate_interpret_artifact(
            artifact=object(),
            target="features",
        )


def test_validate_interpret_artifact_rejects_unknown_backend():
    artifact = _Artifact(backend="unknown")

    with pytest.raises(Edge2TorchError, match="Unsupported artifact backend"):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_accepts_state_update_node_target():
    artifact = _Artifact(backend="state_update")

    _validate_interpret_artifact(
        artifact=artifact,
        target="nodes",
    )


def test_validate_interpret_artifact_rejects_non_list_feature_names():
    artifact = _Artifact(feature_names=("gene_a", "gene_b"))

    with pytest.raises(
        Edge2TorchError,
        match="'artifact.feature_names' must be a list",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_rejects_empty_feature_names():
    artifact = _Artifact(feature_names=[])

    with pytest.raises(
        Edge2TorchError,
        match="at least one feature name",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_rejects_non_string_feature_names():
    artifact = _Artifact(feature_names=["gene_a", 1])

    with pytest.raises(
        Edge2TorchError,
        match="must contain only strings",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_rejects_duplicate_feature_names():
    artifact = _Artifact(feature_names=["gene_a", "gene_a"])

    with pytest.raises(
        Edge2TorchError,
        match="must not contain duplicate names",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_rejects_non_dict_node_names_by_layer():
    artifact = _Artifact(node_names_by_layer=["layer_0"])

    with pytest.raises(
        Edge2TorchError,
        match="node_names_by_layer.*dictionary",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="features",
        )


def test_validate_interpret_artifact_rejects_empty_interpretation_sites():
    artifact = _Artifact(interpretation_sites={})

    with pytest.raises(
        Edge2TorchError,
        match="interpretation_sites.*must not be empty",
    ):
        _validate_interpret_artifact(
            artifact=artifact,
            target="nodes",
        )


# _validate_interpret_data -----------------------------------------------------


def test_validate_interpret_data_rejects_unsupported_data_type():
    with pytest.raises(Edge2TorchError, match="'data' must be"):
        _validate_interpret_data(
            data=[[1.0, 2.0]],
            feature_names=["gene_a", "gene_b"],
        )


def test_validate_interpret_data_accepts_dataframe():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": [3.0, 4.0],
        }
    )

    _validate_interpret_data(
        data=data,
        feature_names=["gene_a", "gene_b"],
    )


def test_validate_interpret_data_accepts_tensor():
    data = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    _validate_interpret_data(
        data=data,
        feature_names=["gene_a", "gene_b"],
    )


# _validate_interpret_dataframe ------------------------------------------------


def test_validate_interpret_dataframe_accepts_integer_columns():
    data = pd.DataFrame({1: [0.1, 0.3], 2: [0.2, 0.4]})

    _validate_interpret_dataframe(
        data=data,
        feature_names=["1", "2"],
        required_features={"1", "2"},
    )


def test_validate_interpret_dataframe_rejects_str_int_column_collision():
    data = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
        ],
        columns=[1, "1", 2],
    )

    with pytest.raises(Edge2TorchError, match="duplicate column names"):
        _validate_interpret_dataframe(
            data=data,
            feature_names=["1", "2"],
            required_features={"1", "2"},
        )


def test_validate_interpret_dataframe_rejects_duplicate_columns():
    data = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        columns=["gene_a", "gene_b", "gene_b"],
    )

    with pytest.raises(Edge2TorchError, match="duplicate column names"):
        _validate_interpret_dataframe(
            data=data,
            feature_names=["gene_a", "gene_b"],
            required_features={"gene_a", "gene_b"},
        )


def test_validate_interpret_dataframe_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
        }
    )

    with pytest.raises(Edge2TorchError, match="missing required feature"):
        _validate_interpret_dataframe(
            data=data,
            feature_names=["gene_a", "gene_b"],
            required_features={"gene_a", "gene_b"},
        )


def test_validate_interpret_dataframe_rejects_extra_columns():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": [3.0, 4.0],
            "gene_c": [5.0, 6.0],
        }
    )

    with pytest.raises(Edge2TorchError, match="not input nodes"):
        _validate_interpret_dataframe(
            data=data,
            feature_names=["gene_a", "gene_b"],
            required_features={"gene_a", "gene_b"},
        )


def test_validate_interpret_dataframe_rejects_non_numeric_columns():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": ["high", "low"],
        }
    )

    with pytest.raises(Edge2TorchError, match="non-numeric feature column"):
        _validate_interpret_dataframe(
            data=data,
            feature_names=["gene_a", "gene_b"],
            required_features={"gene_a", "gene_b"},
        )


# _validate_interpret_anndata --------------------------------------------------


def test_validate_interpret_anndata_accepts_matching_var_names():
    ad = pytest.importorskip("anndata")

    data = ad.AnnData(
        X=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    _validate_interpret_anndata(
        data=data,
        required_features={"gene_a", "gene_b"},
    )


def test_validate_interpret_data_accepts_anndata():
    ad = pytest.importorskip("anndata")

    data = ad.AnnData(
        X=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    _validate_interpret_data(
        data=data,
        feature_names=["gene_a", "gene_b"],
    )


def test_validate_interpret_anndata_rejects_duplicate_var_names():
    ad = pytest.importorskip("anndata")

    with pytest.warns(UserWarning, match="Variable names are not unique"):
        data = ad.AnnData(
            X=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=float,
            ),
            var=pd.DataFrame(index=["gene_a", "gene_a"]),
        )

    with pytest.raises(
        Edge2TorchError,
        match="var_names.*duplicates",
    ):
        _validate_interpret_anndata(
            data=data,
            required_features={"gene_a", "gene_b"},
        )


def test_validate_interpret_anndata_rejects_missing_var_names():
    ad = pytest.importorskip("anndata")

    data = ad.AnnData(
        X=np.array(
            [
                [1.0],
                [2.0],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["gene_a"]),
    )

    with pytest.raises(
        Edge2TorchError,
        match="missing required variable",
    ):
        _validate_interpret_anndata(
            data=data,
            required_features={"gene_a", "gene_b"},
        )


def test_validate_interpret_anndata_rejects_extra_var_names():
    ad = pytest.importorskip("anndata")

    data = ad.AnnData(
        X=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b", "gene_c"]),
    )

    with pytest.raises(
        Edge2TorchError,
        match="not input nodes",
    ):
        _validate_interpret_anndata(
            data=data,
            required_features={"gene_a", "gene_b"},
        )


# _validate_interpret_tensor ---------------------------------------------------


def test_validate_interpret_tensor_rejects_1d_tensor():
    data = torch.tensor([1.0, 2.0])

    with pytest.raises(Edge2TorchError, match="2-dimensional"):
        _validate_interpret_tensor(
            data=data,
            expected_n_features=2,
        )


def test_validate_interpret_tensor_rejects_wrong_width():
    data = torch.randn(2, 3)

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        _validate_interpret_tensor(
            data=data,
            expected_n_features=2,
        )


def test_validate_interpret_tensor_accepts_correct_width():
    data = torch.randn(2, 2)

    _validate_interpret_tensor(
        data=data,
        expected_n_features=2,
    )
