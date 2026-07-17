from typing import cast

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

    ``compile_graph()`` builds a sparse neural network from an edgelist.
    Input nodes are inferred from the graph structure and stored in
    ``artifact.feature_names``. These names define the required column order
    for tensors passed to the compiled PyTorch model.

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
    data : pd.DataFrame | torch.Tensor | anndata.AnnData
        Input data to align. ``AnnData`` is supported when ``anndata`` is
        installed.
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
        extra features are present in named data containers, non-numeric
        DataFrame columns are present, or tensor input has an incompatible
        shape.

    Examples
    --------
    Align a DataFrame whose columns are named but not ordered like the
    compiled model input nodes.

    >>> import pandas as pd
    >>> import torch
    >>> from edge2torch import align_features_to_input_nodes, compile_graph
    >>>
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["feature_a", "feature_b", "hidden"],
    ...         "target": ["hidden", "hidden", "prediction"],
    ...     }
    ... )
    >>> model, artifact = compile_graph(edgelist, quiet=True)
    >>>
    >>> data = pd.DataFrame(
    ...     {
    ...         "feature_b": [2.0, 4.0],
    ...         "feature_a": [1.0, 3.0],
    ...     }
    ... )
    >>>
    >>> artifact.feature_names
    ['feature_a', 'feature_b']
    >>>
    >>> x = align_features_to_input_nodes(
    ...     data=data,
    ...     artifact=artifact,
    ... )
    >>> x
    tensor([[1., 2.],
            [3., 4.]])

    Tensor inputs do not contain feature names, so they are only checked by
    shape and are assumed to already follow ``artifact.feature_names``.

    >>> x_tensor = torch.tensor(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 4.0],
    ...     ]
    ... )
    >>> x_from_tensor = align_features_to_input_nodes(
    ...     data=x_tensor,
    ...     artifact=artifact,
    ... )
    >>> torch.equal(x_from_tensor, x_tensor)
    True
    """
    _validate_align_features_to_input_nodes_inputs(
        data=data,
        artifact=artifact,
    )

    feature_names = list(artifact.feature_names)

    # Dataframe logic
    if isinstance(data, pd.DataFrame):
        ordered = cast(pd.DataFrame, data[feature_names])

        return torch.tensor(
            ordered.to_numpy(copy=True),
            dtype=torch.float32,
        )

    # AnnData logic
    if ad is not None and isinstance(data, ad.AnnData):
        # AnnData stores samples in rows and named variables/features in
        # columns.
        # Reorder data.X columns to match artifact.feature_names.
        var_names = list(data.var_names)
        order = [var_names.index(name) for name in feature_names]
        matrix = data.X[:, order]

        # AnnData.X is often a SciPy sparse matrix; convert it to a dense array
        # before constructing a standard PyTorch tensor.
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        return torch.tensor(
            matrix.copy(),
            dtype=torch.float32,
        )

    # Tensor logic
    if isinstance(data, torch.Tensor):
        # Tensor inputs have no feature names. Validation has already checked
        # that the tensor is 2-dimensional and has the expected number of
        # columns.
        # The column order is assumed to match artifact.feature_names.
        return data.to(dtype=torch.float32)

    raise Edge2TorchError(
        "Unsupported input data type. Expected pandas DataFrame, torch Tensor, "
        "or AnnData if the optional anndata dependency is installed with "
        "'pip install \"edge2torch[anndata]\"'."
    )


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
