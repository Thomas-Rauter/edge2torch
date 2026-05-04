"""
API function
"""

import pandas as pd
import torch

from .compile.artifact import CompileArtifact
from .utils.errors import Edge2TorchError

try:
    import anndata as ad
except ImportError:
    ad = None


def align_features_to_input_nodes(
    data,
    artifact: CompileArtifact,
) -> torch.Tensor:
    """
    Align data features to the input-node order expected by a compiled model.

    ``compile_graph()`` infers model input nodes from the graph structure.
    These input-node names are stored in ``artifact.feature_names`` and define
    the required column order for tensors passed to the compiled PyTorch model.

    For named data containers, this function validates exact feature-name
    compatibility and reorders features by name:

    - ``pandas.DataFrame`` inputs are aligned using column names.
    - ``AnnData`` inputs are aligned using ``var_names`` if ``anndata`` is
      installed.

    Named data containers must contain exactly the compiled model input-node
    features, although they may appear in any order. Missing or extra features
    raise an error.

    ``torch.Tensor`` inputs do not contain feature names, so they are only
    validated by shape and are assumed to already follow
    ``artifact.feature_names`` order.

    Parameters
    ----------
    data
        Input data to align. Supported types are ``pandas.DataFrame``,
        ``torch.Tensor``, and optionally ``anndata.AnnData``.
    artifact : CompileArtifact
        Compilation artifact returned by ``compile_graph()``. Its
        ``feature_names`` field defines the required input-node order.

    Returns
    -------
    torch.Tensor
        Float32 input tensor whose columns are ordered according to
        ``artifact.feature_names``.

    Raises
    ------
    Edge2TorchError
        If the input data type is unsupported, required features are missing,
        extra features are present in named data containers, or tensor input
        has an incompatible shape.

    Examples
    --------
    >>> model, artifact = compile_graph(edgelist)
    >>> x = align_features_to_input_nodes(data_frame, artifact)
    >>> y = model(x)
    """
    _validate_align_features_to_input_nodes_inputs(
        data=data,
        artifact=artifact,
    )

    feature_names = list(artifact.feature_names)

    if isinstance(data, pd.DataFrame):
        ordered = data.loc[:, feature_names]

        return torch.tensor(
            ordered.to_numpy(copy=True),
            dtype=torch.float32,
        )

    if ad is not None and isinstance(data, ad.AnnData):
        var_names = list(data.var_names)
        order = [var_names.index(name) for name in feature_names]
        matrix = data.X[:, order]

        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        return torch.tensor(
            matrix.copy(),
            dtype=torch.float32,
        )

    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)

    raise Edge2TorchError("Unsupported input data type.")


# Level 1 functions (functions called by API functions) ------------------------


def _validate_align_features_to_input_nodes_inputs(
    data,
    artifact,
) -> None:
    """
    Validate the public inputs of ``align_features_to_input_nodes()``.

    Parameters
    ----------
    data
        Input data to align.
    artifact
        Compilation artifact returned by ``compile_graph()``.

    Raises
    ------
    Edge2TorchError
        If any input is invalid.
    """
    if not isinstance(artifact, CompileArtifact):
        raise Edge2TorchError("'artifact' must be a CompileArtifact.")

    feature_names = getattr(artifact, "feature_names", None)

    if not isinstance(feature_names, list):
        raise Edge2TorchError(
            "'artifact.feature_names' must be a list of strings."
        )

    if not feature_names:
        raise Edge2TorchError(
            "The artifact does not define any input-node feature names."
        )

    if not all(isinstance(name, str) for name in feature_names):
        raise Edge2TorchError(
            "'artifact.feature_names' must contain only strings."
        )

    if len(set(feature_names)) != len(feature_names):
        raise Edge2TorchError(
            "'artifact.feature_names' must not contain duplicate names."
        )

    required_features = set(feature_names)

    if isinstance(data, pd.DataFrame):
        if data.columns.duplicated().any():
            raise Edge2TorchError(
                "Input DataFrame must not contain duplicate column names."
            )

        data_features = set(data.columns)

        missing_features = required_features.difference(data_features)
        if missing_features:
            missing_str = ", ".join(sorted(missing_features))
            raise Edge2TorchError(
                "Input data is missing required feature name(s): "
                f"{missing_str}."
            )

        extra_features = data_features.difference(required_features)
        if extra_features:
            extra_str = ", ".join(sorted(extra_features))
            raise Edge2TorchError(
                "Input data contains feature name(s) that are not input nodes "
                f"in the compiled model: {extra_str}."
            )

        non_numeric_features = [
            name
            for name in feature_names
            if not pd.api.types.is_numeric_dtype(data[name])
        ]

        if non_numeric_features:
            non_numeric_str = ", ".join(sorted(non_numeric_features))
            raise Edge2TorchError(
                "Input data contains non-numeric feature column(s): "
                f"{non_numeric_str}."
            )

        return

    if ad is not None and isinstance(data, ad.AnnData):
        var_names = list(data.var_names)

        if len(set(var_names)) != len(var_names):
            raise Edge2TorchError(
                "AnnData var_names must not contain duplicates."
            )

        data_features = set(var_names)

        missing_features = required_features.difference(data_features)
        if missing_features:
            missing_str = ", ".join(sorted(missing_features))
            raise Edge2TorchError(
                f"AnnData is missing required feature name(s): {missing_str}."
            )

        extra_features = data_features.difference(required_features)
        if extra_features:
            extra_str = ", ".join(sorted(extra_features))
            raise Edge2TorchError(
                "AnnData contains feature name(s) that are not input nodes "
                f"in the compiled model: {extra_str}."
            )

        return

    if isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise Edge2TorchError("Input tensor must be 2-dimensional.")

        expected_n_features = len(feature_names)

        if data.shape[1] != expected_n_features:
            raise Edge2TorchError(
                "Input tensor has the wrong number of features. "
                f"Expected {expected_n_features}, got {data.shape[1]}."
            )

        return

    raise Edge2TorchError(
        "Unsupported input data type. Expected pandas DataFrame, "
        "torch Tensor, or AnnData if anndata is installed."
    )
