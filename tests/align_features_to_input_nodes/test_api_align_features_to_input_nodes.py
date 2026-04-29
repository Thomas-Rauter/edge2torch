import pandas as pd
import pytest
import torch

from kpnn.align_features_to_input_nodes import align_features_to_input_nodes
from kpnn.compile.artifact import KPNNArtifact
from kpnn.compile_graph import compile_graph
from kpnn.utils.errors import KPNNError


def _compile_simple_artifact() -> KPNNArtifact:
    edgelist = pd.DataFrame(
        {
            "source": ["gene_a", "gene_b"],
            "target": ["pathway_1", "pathway_1"],
        }
    )

    _, artifact = compile_graph(edgelist, quiet=True)

    return artifact


def test_align_features_to_input_nodes_reorders_dataframe_columns():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_b": [2.0, 4.0],
            "gene_a": [1.0, 3.0],
        }
    )

    result = align_features_to_input_nodes(data, artifact)

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)


def test_align_features_to_input_nodes_rejects_missing_dataframe_features():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
        }
    )

    with pytest.raises(KPNNError, match="missing required feature"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_extra_dataframe_features():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": [3.0, 4.0],
            "gene_c": [5.0, 6.0],
        }
    )

    with pytest.raises(KPNNError, match="not input nodes"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_duplicate_dataframe_columns():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        columns=["gene_a", "gene_b", "gene_b"],
    )

    with pytest.raises(KPNNError, match="duplicate column names"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_non_numeric_dataframe_columns():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": ["high", "low"],
        }
    )

    with pytest.raises(KPNNError, match="non-numeric feature column"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_accepts_correct_tensor_shape():
    artifact = _compile_simple_artifact()

    data = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float64,
    )

    result = align_features_to_input_nodes(data, artifact)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.equal(
        result,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_align_features_to_input_nodes_rejects_1d_tensor():
    artifact = _compile_simple_artifact()

    data = torch.tensor([1.0, 2.0])

    with pytest.raises(KPNNError, match="2-dimensional"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_tensor_with_wrong_width():
    artifact = _compile_simple_artifact()

    data = torch.randn(3, 3)

    with pytest.raises(KPNNError, match="wrong number of features"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_invalid_artifact_type():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": [3.0, 4.0],
        }
    )

    with pytest.raises(KPNNError, match="must be a KPNNArtifact"):
        align_features_to_input_nodes(data, artifact=object())


def test_align_features_to_input_nodes_rejects_artifact_without_feature_names():
    artifact = KPNNArtifact(
        backend="feedforward",
        graph=object(),  # type: ignore[arg-type]
        execution_plan=object(),
        node_names_by_layer={},
        feature_names=[],
    )

    data = pd.DataFrame()

    with pytest.raises(KPNNError, match="does not define any input-node"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_duplicate_artifact_features():
    artifact = KPNNArtifact(
        backend="feedforward",
        graph=object(),  # type: ignore[arg-type]
        execution_plan=object(),
        node_names_by_layer={},
        feature_names=["gene_a", "gene_a"],
    )

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
        }
    )

    with pytest.raises(KPNNError, match="must not contain duplicate"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_unsupported_data_type():
    artifact = _compile_simple_artifact()

    with pytest.raises(KPNNError, match="Unsupported input data type"):
        align_features_to_input_nodes([[1.0, 2.0]], artifact)
