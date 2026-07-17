from typing import Any, Union

import pandas as pd

from .interpret.captum_adapter import run_captum_interpretation
from .interpret.input_validation import validate_interpret_model_inputs
from .interpret.prepare_input import prepare_interpretation_input


def interpret_model(
    model: Any,
    artifact: Any,
    data: Any,
    target: str = "features",
    method: str = "IntegratedGradients",
    constructor_kwargs: dict[str, Any] | None = None,
    attribute_kwargs: dict[str, Any] | None = None,
    level: str = "summary",
    nodes: str = "hidden",
    site_aggregation: str = "max_abs",
    quiet: bool = False,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Interpret a model compiled by edge2torch using a Captum attribution
    method.

    Parameters
    ----------
    model : Any
        PyTorch model returned by ``compile_graph()``, optionally customized
        and trained by the user.
    artifact : Any
        Compilation artifact returned by ``compile_graph()``.
    data : pd.DataFrame | anndata.AnnData | torch.Tensor
        Input data used for attribution.
    target : str, default="features"
        Interpretation target. Use ``"features"`` to attribute predictions
        to input features. Use ``"nodes"`` to attribute predictions to named
        graph nodes.
    method : str, default="IntegratedGradients"
        Captum attribution method name. Method names follow Captum class
        names exactly and are case-sensitive, for example
        ``"IntegratedGradients"``, ``"Saliency"``, ``"DeepLift"``,
        ``"LayerConductance"``, or ``"LayerIntegratedGradients"``.

        The selected method must be compatible with ``target`` and the
        compiled backend. If an unsupported method is provided, edge2torch
        raises an error listing the supported method names.
    constructor_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed directly to the constructor of the
        selected Captum attribution class. These arguments are passed through
        unchanged and are not interpreted, validated, or modified by
        edge2torch. Refer to the Captum documentation for the selected method
        to determine which constructor arguments are supported.
    attribute_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed directly to the selected Captum
        method's ``attribute()`` call. These arguments are passed through
        unchanged and are not interpreted, validated, or modified by
        edge2torch. Refer to the Captum documentation for the selected method
        to determine which attribution arguments are supported.
    level : str, default="summary"
        Detail level for ``target="nodes"``. Use ``"summary"`` to return one
        node-importance table per sample. Use ``"sites"`` to return one table
        per interpretation site such as ``layer_1`` or ``step_2``.
    nodes : str, default="hidden"
        Node filter for ``target="nodes"``. Use ``"hidden"`` for internal
        graph nodes, ``"non_input"`` to include output nodes, or ``"all"``
        for all visible graph nodes.
    site_aggregation : str, default="max_abs"
        Aggregation rule used when ``target="nodes"`` and ``level="summary"``
        for state_update backends. Ignored for feedforward summary
        results and for ``level="sites"``.
    quiet : bool, default=False
        If False, emit informational notes. If True, suppress informational
        notes.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        If ``target="features"``, returns one DataFrame with rows as
        examples and columns as input feature names.

        If ``target="nodes"`` and ``level="summary"``, returns one DataFrame
        with rows as examples and columns as named nodes selected by
        ``nodes``.

        If ``target="nodes"`` and ``level="sites"``, returns a dictionary
        mapping site identifiers to DataFrames. Each DataFrame has rows as
        examples and columns as named nodes for that site.

    Notes
    -----
    Feature interpretation is supported for all implemented backends.

    Node interpretation is supported for the ``feedforward`` and
    ``state_update`` backends. Node interpretation methods use Captum layer
    attribution classes.

    For node-level interpretation, edge2torch must access the compiled
    model's internal interpretation sites. This works for raw models
    returned by ``compile_graph()``, models returned by ``customize_model()``,
    and manually wrapped PyTorch models if the compiled model remains
    registered as a PyTorch submodule. Highly custom wrappers that hide,
    replace, or bypass the compiled model may not support ``target="nodes"``.

    ``interpret_model()`` temporarily switches the model to evaluation mode
    while computing attributions and restores the previous training/evaluation
    mode afterward.

    ``constructor_kwargs`` and ``attribute_kwargs`` are passed through to
    Captum. Refer to the Captum documentation for method-specific arguments
    such as baselines, targets, additional forward arguments, or perturbation
    settings.

    Raises
    ------
    Edge2TorchError
        If interpretation input validation fails, the requested
        target / method / backend combination is not supported, or Captum
        returns unsupported output.

    Examples
    --------
    Compute feature-level attributions with integrated gradients.

    >>> feature_attributions = interpret_model(
    ...     model=trained_model,
    ...     artifact=artifact,
    ...     data=data,
    ...     target="features",
    ...     method="IntegratedGradients",
    ...     quiet=True,
    ... )
    >>> feature_attributions.head()

    Compute summary node-level attributions.

    >>> node_importance = interpret_model(
    ...     model=trained_model,
    ...     artifact=artifact,
    ...     data=data,
    ...     target="nodes",
    ...     method="LayerConductance",
    ...     quiet=True,
    ... )
    >>> node_importance.head()

    Compute per-site node-level attributions.

    >>> node_attributions_by_site = interpret_model(
    ...     model=trained_model,
    ...     artifact=artifact,
    ...     data=data,
    ...     target="nodes",
    ...     level="sites",
    ...     nodes="non_input",
    ...     method="LayerConductance",
    ...     quiet=True,
    ... )
    >>> node_attributions_by_site.keys()
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
        level=level,
        nodes=nodes,
        site_aggregation=site_aggregation,
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
        level=level,
        nodes=nodes,
        site_aggregation=site_aggregation,
    )

    if not quiet:
        print(
            f"[edge2torch] Finished interpretation with method '{method}' "
            f"for target '{target}'."
        )

    return result
