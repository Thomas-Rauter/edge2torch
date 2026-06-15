"""
Validation logic for the interpret_model() public API.

Why this file exists
--------------------
This file separates strict public input validation from both the
interpret_model() API wrapper and the internal interpretation execution
logic. Keeping validation here makes the supported interpretation
contract easier to reason about and avoids duplicating checks across
input preparation and Captum-specific modules.

Role in the package
-------------------
This is an internal validation module for model interpretation. It
defines which target / method / backend / data combinations are accepted
by interpret_model() and raises clear errors for unsupported or
ambiguous usage. It should not contain Captum execution logic, input
standardization, or public API orchestration.
"""

import pandas as pd
import torch
from torch import nn

try:
    import anndata as ad
except ImportError:
    ad = None

from ..utils.errors import Edge2TorchError
from .method_registry import (
    FEATURE_METHODS,
    FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS,
    NODE_METHODS,
    NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS,
    SUPPORTED_METHODS,
)

# Level 1 functions (called by API functions) ----------------------------------


def validate_interpret_model_inputs(
    model,
    artifact,
    data,
    target,
    method,
    constructor_kwargs,
    attribute_kwargs,
    quiet,
    level="summary",
    nodes="hidden",
    site_aggregation="max_abs",
) -> None:
    """
    Validate the public inputs of ``interpret_model()``.

    Parameters
    ----------
    model
        Trained PyTorch model returned by ``compile_graph()``.
    artifact
        Compilation artifact returned by ``compile_graph()``.
    data
        Input data used for attribution.
    target
        Interpretation target.
    method
        Captum attribution method.
    constructor_kwargs
        Optional keyword arguments passed to the Captum attribution class
        constructor.
    attribute_kwargs
        Optional keyword arguments passed to the Captum ``attribute()`` call.
    quiet
        Whether informational notes should be suppressed.
    level
        Node interpretation detail level.
    nodes
        Node filter for ``target="nodes"``.
    site_aggregation
        Aggregation rule for summary node interpretation in recurrent and
        graphnn backends.

    Raises
    ------
    Edge2TorchError
        If any input is invalid.
    """
    _validate_interpret_options(
        target=target,
        method=method,
        quiet=quiet,
        constructor_kwargs=constructor_kwargs,
        attribute_kwargs=attribute_kwargs,
        level=level,
        nodes=nodes,
        site_aggregation=site_aggregation,
    )

    _validate_interpret_model(model=model)

    _validate_interpret_artifact(
        artifact=artifact,
        target=target,
    )

    _validate_interpret_data(
        data=data,
        feature_names=list(artifact.feature_names),
    )


# Level 2 functions (called by level 1 functions) ------------------------------


def _validate_interpret_options(
    target,
    method,
    quiet,
    constructor_kwargs,
    attribute_kwargs,
    level="summary",
    nodes="hidden",
    site_aggregation="max_abs",
) -> None:
    """
    Validate interpretation target, method, verbosity, and Captum kwargs.
    """
    if not isinstance(quiet, bool):
        raise Edge2TorchError(
            "'quiet' must be a boolean value (True or False)."
        )

    if not isinstance(target, str):
        raise Edge2TorchError("'target' must be a string.")

    supported_targets = {"nodes", "features"}
    if target not in supported_targets:
        supported = ", ".join(sorted(supported_targets))
        raise Edge2TorchError(
            f"Unsupported target '{target}'. Expected one of: {supported}."
        )

    if not isinstance(method, str):
        raise Edge2TorchError("'method' must be a string.")

    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise Edge2TorchError(
            f"Unsupported method '{method}'. Expected one of: {supported}."
        )

    if target == "features" and method not in FEATURE_METHODS:
        raise Edge2TorchError(
            f"Method '{method}' is not compatible with target='features'."
        )

    if target == "nodes" and method not in NODE_METHODS:
        raise Edge2TorchError(
            f"Method '{method}' is not compatible with target='nodes'."
        )

    if target == "nodes":
        _validate_node_interpretation_options(
            level=level,
            nodes=nodes,
            site_aggregation=site_aggregation,
        )

    if constructor_kwargs is not None and not isinstance(
        constructor_kwargs,
        dict,
    ):
        raise Edge2TorchError(
            "'constructor_kwargs' must be a dictionary or None."
        )

    if (
        target == "features"
        and method in FEATURE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS
        and constructor_kwargs
    ):
        raise Edge2TorchError(
            f"Method '{method}' does not support constructor_kwargs. "
            "Pass method-specific attribution options via attribute_kwargs."
        )

    if (
        target == "nodes"
        and method in NODE_METHODS_WITHOUT_CONSTRUCTOR_KWARGS
        and constructor_kwargs
    ):
        raise Edge2TorchError(
            f"Method '{method}' does not support constructor_kwargs. "
            "Pass method-specific attribution options via attribute_kwargs."
        )

    if attribute_kwargs is not None and not isinstance(attribute_kwargs, dict):
        raise Edge2TorchError(
            "'attribute_kwargs' must be a dictionary or None."
        )

    if (
        isinstance(attribute_kwargs, dict)
        and attribute_kwargs.get("return_convergence_delta") is True
    ):
        raise Edge2TorchError(
            "'return_convergence_delta=True' is not currently supported "
            "because edge2torch expects Captum attribution methods to return "
            "attribution tensors."
        )


def _validate_interpret_model(model) -> None:
    """
    Validate the model object.
    """
    if not isinstance(model, nn.Module):
        raise Edge2TorchError("'model' must be a torch.nn.Module.")

    if not hasattr(model, "forward"):
        raise Edge2TorchError(
            "'model' must be a PyTorch model with a forward method."
        )


def _validate_interpret_artifact(
    artifact,
    target,
) -> None:
    """
    Validate the compilation artifact and backend compatibility.
    """
    supported_backends = {
        "feedforward",
        "recurrent",
        "graphnn",
    }

    required_artifact_attrs = {
        "backend",
        "feature_names",
        "node_names_by_layer",
        "interpretation_sites",
        "input_nodes",
        "output_nodes",
        "hidden_nodes",
        "execution_plan",
    }
    missing_attrs = [
        attr for attr in required_artifact_attrs if not hasattr(artifact, attr)
    ]

    if missing_attrs:
        missing_str = ", ".join(sorted(missing_attrs))
        raise Edge2TorchError(
            f"'artifact' is missing required attribute(s): {missing_str}."
        )

    if artifact.backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise Edge2TorchError(
            f"Unsupported artifact backend '{artifact.backend}'. "
            f"Expected one of: {supported}."
        )

    if target == "nodes" and not isinstance(
        artifact.interpretation_sites, dict
    ):
        raise Edge2TorchError(
            "'artifact.interpretation_sites' must be a dictionary."
        )

    if target == "nodes" and not artifact.interpretation_sites:
        raise Edge2TorchError(
            "'artifact.interpretation_sites' must not be empty for "
            "node interpretation."
        )

    if target == "nodes" and not isinstance(artifact.input_nodes, list):
        raise Edge2TorchError("'artifact.input_nodes' must be a list.")

    if target == "nodes" and not isinstance(artifact.output_nodes, list):
        raise Edge2TorchError("'artifact.output_nodes' must be a list.")

    if target == "nodes" and not isinstance(artifact.hidden_nodes, list):
        raise Edge2TorchError("'artifact.hidden_nodes' must be a list.")

    if not isinstance(artifact.feature_names, list):
        raise Edge2TorchError("'artifact.feature_names' must be a list.")

    if not artifact.feature_names:
        raise Edge2TorchError(
            "'artifact.feature_names' must contain at least one feature name."
        )

    if not all(isinstance(name, str) for name in artifact.feature_names):
        raise Edge2TorchError(
            "'artifact.feature_names' must contain only strings."
        )

    if len(set(artifact.feature_names)) != len(artifact.feature_names):
        raise Edge2TorchError(
            "'artifact.feature_names' must not contain duplicate names."
        )

    if not isinstance(artifact.node_names_by_layer, dict):
        raise Edge2TorchError(
            "'artifact.node_names_by_layer' must be a dictionary."
        )


def _validate_node_interpretation_options(
    level,
    nodes,
    site_aggregation,
) -> None:
    """
    Validate node-interpretation detail options.
    """
    supported_levels = {"sites", "summary"}
    if level not in supported_levels:
        supported = ", ".join(sorted(supported_levels))
        raise Edge2TorchError(
            f"Unsupported level '{level}'. Expected one of: {supported}."
        )

    supported_node_filters = {"hidden", "all", "non_input"}
    if nodes not in supported_node_filters:
        supported = ", ".join(sorted(supported_node_filters))
        raise Edge2TorchError(
            f"Unsupported nodes filter '{nodes}'. Expected one of: {supported}."
        )

    supported_aggregations = {"max_abs", "mean_abs", "last"}
    if site_aggregation not in supported_aggregations:
        supported = ", ".join(sorted(supported_aggregations))
        raise Edge2TorchError(
            f"Unsupported site_aggregation '{site_aggregation}'. "
            f"Expected one of: {supported}."
        )


def _validate_interpret_data(
    data,
    feature_names: list[str],
) -> None:
    """
    Validate interpretation input data against expected feature names.
    """
    is_supported_data = isinstance(data, (pd.DataFrame, torch.Tensor))

    if ad is not None:
        is_supported_data = is_supported_data or isinstance(data, ad.AnnData)

    if not is_supported_data:
        if ad is None:
            raise Edge2TorchError(
                "'data' must be a pandas DataFrame or torch.Tensor. "
                "AnnData support requires the optional 'anndata' dependency."
            )

        raise Edge2TorchError(
            "'data' must be a pandas DataFrame, anndata.AnnData, "
            "or torch.Tensor."
        )

    required_features = set(feature_names)
    expected_n_features = len(feature_names)

    if isinstance(data, pd.DataFrame):
        _validate_interpret_dataframe(
            data=data,
            feature_names=feature_names,
            required_features=required_features,
        )
        return

    if ad is not None and isinstance(data, ad.AnnData):
        _validate_interpret_anndata(
            data=data,
            required_features=required_features,
        )
        return

    if isinstance(data, torch.Tensor):
        _validate_interpret_tensor(
            data=data,
            expected_n_features=expected_n_features,
        )
        return

    raise Edge2TorchError("Unsupported interpretation input type.")


# Level 3 functions (called by level 2 functions) ------------------------------


def _validate_interpret_dataframe(
    data: pd.DataFrame,
    feature_names: list[str],
    required_features: set[str],
) -> None:
    """
    Validate DataFrame interpretation input.
    """
    if data.columns.duplicated().any():
        raise Edge2TorchError(
            "'data' DataFrame must not contain duplicate column names."
        )

    data_features = set(data.columns)

    missing_columns = required_features.difference(data_features)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise Edge2TorchError(
            f"'data' is missing required feature column(s): {missing_str}."
        )

    extra_columns = data_features.difference(required_features)
    if extra_columns:
        extra_str = ", ".join(sorted(extra_columns))
        raise Edge2TorchError(
            "'data' contains feature column(s) that are not input nodes "
            f"in the compiled model: {extra_str}."
        )

    non_numeric_columns = [
        name
        for name in feature_names
        if not pd.api.types.is_numeric_dtype(data[name])
    ]

    if non_numeric_columns:
        non_numeric_str = ", ".join(sorted(non_numeric_columns))
        raise Edge2TorchError(
            f"'data' contains non-numeric feature column(s): {non_numeric_str}."
        )


def _validate_interpret_anndata(
    data,
    required_features: set[str],
) -> None:
    """
    Validate AnnData interpretation input.
    """
    var_names = list(data.var_names)

    if len(set(var_names)) != len(var_names):
        raise Edge2TorchError("'data.var_names' must not contain duplicates.")

    data_features = set(var_names)

    missing_vars = required_features.difference(data_features)
    if missing_vars:
        missing_str = ", ".join(sorted(missing_vars))
        raise Edge2TorchError(
            f"'data' is missing required variable name(s): {missing_str}."
        )

    extra_vars = data_features.difference(required_features)
    if extra_vars:
        extra_str = ", ".join(sorted(extra_vars))
        raise Edge2TorchError(
            "'data' contains variable name(s) that are not input nodes "
            f"in the compiled model: {extra_str}."
        )


def _validate_interpret_tensor(
    data: torch.Tensor,
    expected_n_features: int,
) -> None:
    """
    Validate tensor interpretation input.
    """
    if data.ndim != 2:
        raise Edge2TorchError("'data' tensor must be 2-dimensional.")

    if data.shape[1] != expected_n_features:
        raise Edge2TorchError(
            "'data' tensor has the wrong number of features. "
            f"Expected {expected_n_features}, got {data.shape[1]}."
        )
