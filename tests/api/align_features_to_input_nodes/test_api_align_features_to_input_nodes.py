import numpy as np
import pandas as pd
import pytest
import torch

from edge2torch.align_features_to_input_nodes import (
    align_features_to_input_nodes,
)
from edge2torch.compile.artifact import CompileArtifact
from edge2torch.compile_graph import compile_graph
from edge2torch.utils.errors import Edge2TorchError


def _compile_simple_artifact() -> CompileArtifact:
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


def test_align_features_to_input_nodes_accepts_integer_dataframe_columns():
    edgelist = pd.DataFrame({"source": [1, 2], "target": [3, 3]})
    _, artifact = compile_graph(edgelist, quiet=True)

    assert artifact.feature_names == ["1", "2"]

    data = pd.DataFrame({1: [0.1, 0.3], 2: [0.2, 0.4]})
    result = align_features_to_input_nodes(data, artifact)

    expected = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected)


def test_align_features_to_input_nodes_accepts_numpy_int_columns():
    edgelist = pd.DataFrame({"source": [1, 2], "target": [3, 3]})
    _, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        {
            np.int64(1): [0.1],
            np.int64(2): [0.2],
        }
    )
    result = align_features_to_input_nodes(data, artifact)

    expected = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_align_features_to_input_nodes_rejects_str_int_column_collision():
    edgelist = pd.DataFrame({"source": [1, 2], "target": [3, 3]})
    _, artifact = compile_graph(edgelist, quiet=True)

    data = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
        ],
        columns=[1, "1", 2],
    )

    with pytest.raises(Edge2TorchError, match="duplicate column names"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_reorders_numeric_string_anndata_vars():
    # AnnData stores var_names as strings; use string IDs that match the
    # stringified integer node names from compile_graph().
    ad = pytest.importorskip("anndata")

    edgelist = pd.DataFrame({"source": [1, 2], "target": [3, 3]})
    _, artifact = compile_graph(edgelist, quiet=True)

    data = ad.AnnData(
        X=np.array([[0.2, 0.1]], dtype=float),
        var=pd.DataFrame(index=["2", "1"]),
    )
    result = align_features_to_input_nodes(data, artifact)

    expected = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_align_features_to_input_nodes_rejects_missing_dataframe_features():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
        }
    )

    with pytest.raises(Edge2TorchError, match="missing required feature"):
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

    with pytest.raises(Edge2TorchError, match="not input nodes"):
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

    with pytest.raises(Edge2TorchError, match="duplicate column names"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_non_numeric_dataframe_columns():
    artifact = _compile_simple_artifact()

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": ["high", "low"],
        }
    )

    with pytest.raises(Edge2TorchError, match="non-numeric feature column"):
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

    with pytest.raises(Edge2TorchError, match="2-dimensional"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_tensor_with_wrong_width():
    artifact = _compile_simple_artifact()

    data = torch.randn(3, 3)

    with pytest.raises(Edge2TorchError, match="wrong number of features"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_invalid_artifact_type():
    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
            "gene_b": [3.0, 4.0],
        }
    )

    with pytest.raises(Edge2TorchError, match="must be a CompileArtifact"):
        align_features_to_input_nodes(data, artifact=object())


def test_align_features_to_input_nodes_rejects_artifact_without_feature_names():
    artifact = CompileArtifact(
        backend="feedforward",
        graph=object(),  # type: ignore[arg-type]
        execution_plan=object(),
        node_names_by_layer={},
        input_nodes=[],
        output_nodes=[],
        hidden_nodes=[],
        interpretation_sites={},
        feature_names=[],
    )

    data = pd.DataFrame()

    with pytest.raises(Edge2TorchError, match="does not define any input-node"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_duplicate_artifact_features():
    artifact = CompileArtifact(
        backend="feedforward",
        graph=object(),  # type: ignore[arg-type]
        execution_plan=object(),
        node_names_by_layer={},
        input_nodes=["gene_a"],
        output_nodes=[],
        hidden_nodes=[],
        interpretation_sites={},
        feature_names=["gene_a", "gene_a"],
    )

    data = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0],
        }
    )

    with pytest.raises(Edge2TorchError, match="must not contain duplicate"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_unsupported_data_type():
    artifact = _compile_simple_artifact()

    with pytest.raises(Edge2TorchError, match="Unsupported input data type"):
        align_features_to_input_nodes([[1.0, 2.0]], artifact)


def test_align_features_to_input_nodes_reorders_anndata_vars():
    ad = pytest.importorskip("anndata")

    artifact = _compile_simple_artifact()

    data = ad.AnnData(
        X=np.array(
            [
                [2.0, 1.0],
                [4.0, 3.0],
            ],
            dtype=float,
        ),
        var=pd.DataFrame(index=["gene_b", "gene_a"]),
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


def test_align_features_to_input_nodes_accepts_sparse_anndata_matrix():
    ad = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")

    artifact = _compile_simple_artifact()

    data = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [2.0, 1.0],
                [4.0, 3.0],
            ]
        ),
        var=pd.DataFrame(index=["gene_b", "gene_a"]),
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


def test_align_features_to_input_nodes_rejects_duplicate_anndata_vars():
    ad = pytest.importorskip("anndata")

    artifact = _compile_simple_artifact()

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

    with pytest.raises(Edge2TorchError, match="var_names must not contain"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_missing_anndata_vars():
    ad = pytest.importorskip("anndata")

    artifact = _compile_simple_artifact()

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

    with pytest.raises(Edge2TorchError, match="missing required feature"):
        align_features_to_input_nodes(data, artifact)


def test_align_features_to_input_nodes_rejects_extra_anndata_vars():
    ad = pytest.importorskip("anndata")

    artifact = _compile_simple_artifact()

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

    with pytest.raises(Edge2TorchError, match="not input nodes"):
        align_features_to_input_nodes(data, artifact)
