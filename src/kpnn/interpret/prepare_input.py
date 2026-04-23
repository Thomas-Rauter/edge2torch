from dataclasses import dataclass

import pandas as pd
import torch

try:
    import anndata as ad
except ImportError:
    ad = None

from ..utils.errors import KPNNError


@dataclass
class PreparedInterpretationInput:
    """
    Standardized interpretation input passed to the Captum adapter.
    """

    inputs: torch.Tensor
    sample_names: list[str]
    feature_names: list[str]


def prepare_interpretation_input(
    data,
    artifact,
) -> PreparedInterpretationInput:
    """
    Convert interpretation input data into a standardized tensor form.

    Parameters
    ----------
    data
        Input data for attribution.
    artifact
        Compilation artifact returned by ``compile_graph()``.

    Returns
    -------
    PreparedInterpretationInput
        Standardized tensor input plus sample and feature names.

    Raises
    ------
    KPNNError
        If input preparation fails.
    """
    feature_names = list(artifact.feature_names)

    if isinstance(data, pd.DataFrame):
        ordered = data.loc[:, feature_names]
        inputs = torch.tensor(
            ordered.to_numpy(),
            dtype=torch.float32,
        )
        sample_names = [str(idx) for idx in ordered.index]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    if ad is not None and isinstance(data, ad.AnnData):
        var_names = list(data.var_names)

        missing_features = set(feature_names).difference(var_names)
        if missing_features:
            missing_str = ", ".join(sorted(missing_features))
            raise KPNNError(
                f"AnnData is missing required feature name(s): {missing_str}."
            )

        order = [var_names.index(name) for name in feature_names]
        matrix = data.X[:, order]

        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        inputs = torch.tensor(
            matrix,
            dtype=torch.float32,
        )
        sample_names = [str(name) for name in data.obs_names]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    if isinstance(data, torch.Tensor):
        inputs = data.to(dtype=torch.float32)
        sample_names = [str(i) for i in range(inputs.shape[0])]

        return PreparedInterpretationInput(
            inputs=inputs,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    raise KPNNError("Unsupported interpretation input type.")
