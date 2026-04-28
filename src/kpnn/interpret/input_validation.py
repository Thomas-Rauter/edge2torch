import pandas as pd

try:
    import anndata as ad
except ImportError:
    ad = None
import torch

from ..utils.errors import KPNNError


def validate_interpret_model_inputs(
    model,
    artifact,
    data,
    target,
    method,
    quiet,
):
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
    quiet
        Whether informational notes should be suppressed.

    Raises
    ------
    KPNNError
        If any input is invalid.
    """
    supported_targets = {"nodes", "features"}
    supported_methods = {
        "integrated_gradients",
        "layer_conductance",
        "layer_integrated_gradients",
    }
    supported_backends = {
        "feedforward",
        "recurrent",
        "graphnn",
    }

    feature_methods = {"integrated_gradients"}
    node_methods = {
        "layer_conductance",
        "layer_integrated_gradients",
    }

    if not isinstance(quiet, bool):
        raise KPNNError(
            "'quiet' must be a boolean value (True or False)."
        )

    if target not in supported_targets:
        supported = ", ".join(sorted(supported_targets))
        raise KPNNError(
            f"Unsupported target '{target}'. Expected one of: "
            f"{supported}."
        )

    if method not in supported_methods:
        supported = ", ".join(sorted(supported_methods))
        raise KPNNError(
            f"Unsupported method '{method}'. Expected one of: "
            f"{supported}."
        )

    if target == "features" and method not in feature_methods:
        raise KPNNError(
            f"Method '{method}' is not compatible with "
            "target='features'."
        )

    if target == "nodes" and method not in node_methods:
        raise KPNNError(
            f"Method '{method}' is not compatible with "
            "target='nodes'."
        )

    if not hasattr(model, "forward"):
        raise KPNNError(
            "'model' must be a PyTorch model with a forward method."
        )

    required_artifact_attrs = {
        "backend",
        "feature_names",
        "node_names_by_layer",
        "execution_plan",
    }
    missing_attrs = [
        attr
        for attr in required_artifact_attrs
        if not hasattr(artifact, attr)
    ]

    if missing_attrs:
        missing_str = ", ".join(sorted(missing_attrs))
        raise KPNNError(
            "'artifact' is missing required attribute(s): "
            f"{missing_str}."
        )

    if artifact.backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise KPNNError(
            f"Unsupported artifact backend '{artifact.backend}'. "
            f"Expected one of: {supported}."
        )

    if target == "nodes" and artifact.backend != "feedforward":
        raise KPNNError(
            "Node interpretation currently supports only the "
            "'feedforward' backend."
        )

    if not isinstance(artifact.feature_names, list):
        raise KPNNError(
            "'artifact.feature_names' must be a list."
        )

    if not all(isinstance(name, str) for name in artifact.feature_names):
        raise KPNNError(
            "'artifact.feature_names' must contain only strings."
        )

    if not isinstance(artifact.node_names_by_layer, dict):
        raise KPNNError(
            "'artifact.node_names_by_layer' must be a dictionary."
        )

    supported_data_types = (pd.DataFrame, torch.Tensor)
    if ad is not None:
        supported_data_types += ad.AnnData,

    if not isinstance(data, supported_data_types):
        if ad is None:
            raise KPNNError(
                "'data' must be a pandas DataFrame or torch.Tensor. "
                "AnnData support requires the optional 'anndata' "
                "dependency."
            )

        raise KPNNError(
            "'data' must be a pandas DataFrame, anndata.AnnData, "
            "or torch.Tensor."
        )

    expected_n_features = len(artifact.feature_names)

    if isinstance(data, pd.DataFrame):
        missing_columns = set(artifact.feature_names).difference(
            data.columns
        )

        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise KPNNError(
                "'data' is missing required feature column(s): "
                f"{missing_str}."
            )

    elif ad is not None and isinstance(data, ad.AnnData):
        if data.n_vars != expected_n_features:
            raise KPNNError(
                "'data' has the wrong number of variables. "
                f"Expected {expected_n_features}, got "
                f"{data.n_vars}."
            )

    elif isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise KPNNError(
                "'data' tensor must be 2-dimensional."
            )

        if data.shape[1] != expected_n_features:
            raise KPNNError(
                "'data' tensor has the wrong number of features. "
                f"Expected {expected_n_features}, got "
                f"{data.shape[1]}."
            )
