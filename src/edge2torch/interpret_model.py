"""
API function
"""

from typing import Any, Union

import pandas as pd

from .interpret.captum_adapter import run_captum_interpretation
from .interpret.input_validation import validate_interpret_model_inputs
from .interpret.prepare_input import prepare_interpretation_input


def interpret_model(
    model: Any,
    artifact: Any,
    data: Any,
    target: str = "nodes",
    method: str = "layer_conductance",
    constructor_kwargs: dict[str, Any] | None = None,
    attribute_kwargs: dict[str, Any] | None = None,
    quiet: bool = False,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Interpret a compiled KPNN model with a Captum-based method.

    Parameters
    ----------
    model : Any
        Trained PyTorch model returned by ``compile_graph()``.
    artifact : Any
        Compilation artifact returned by ``compile_graph()``.
    data : pd.DataFrame | anndata.AnnData | torch.Tensor
        Input data used for attribution.
    target : str, default="nodes"
        Interpretation target. One of: ``"nodes"``, ``"features"``.
    method : str, default="layer_conductance"
        Attribution method. Must be compatible with ``target`` and the
        compiled backend.
    constructor_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed directly to the selected Captum
        attribution class constructor. These arguments are method-specific and
        are not interpreted by edge2torch.
    attribute_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed directly to the selected Captum
        method's ``attribute()`` call. These arguments are method-specific and
        are not interpreted by edge2torch.
    quiet : bool, default=False
        If False, emit informational notes. If True, suppress notes and
        only surface warnings and errors.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        If ``target="features"``, returns one DataFrame with rows as
        examples and columns as feature names.

        If ``target="nodes"``, returns a dictionary mapping layer names to
        DataFrames. Each DataFrame has rows as examples and columns as node
        names for that layer.

    Notes
    -----
    Feature interpretation is currently supported for all implemented
    backends.

    Node interpretation is currently supported only for the
    ``"feedforward"`` backend.

    Raises
    ------
    Edge2TorchError
        If interpretation input validation fails, the requested
        target / method / backend combination is not supported, or Captum
        returns unsupported output.
    """
    validate_interpret_model_inputs(
        model=model,
        artifact=artifact,
        data=data,
        target=target,
        method=method,
        constructor_kwargs=constructor_kwargs,
        attribute_kwargs=attribute_kwargs,
        quiet=quiet,
    )

    constructor_kwargs = (
        {} if constructor_kwargs is None else dict(constructor_kwargs)
    )
    attribute_kwargs = (
        {} if attribute_kwargs is None else dict(attribute_kwargs)
    )

    prepared_input = prepare_interpretation_input(
        data=data,
        artifact=artifact,
    )

    result = run_captum_interpretation(
        model=model,
        artifact=artifact,
        inputs=prepared_input.inputs,
        sample_names=prepared_input.sample_names,
        feature_names=prepared_input.feature_names,
        target=target,
        method=method,
        constructor_kwargs=constructor_kwargs,
        attribute_kwargs=attribute_kwargs,
    )

    if not quiet:
        print(
            f"[edge2torch] Finished interpretation with method '{method}' "
            f"for target '{target}'."
        )

    return result
