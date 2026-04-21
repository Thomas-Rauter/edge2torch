import pandas as pd

from .errors import KPNNError


def validate_compile_graph_inputs(
    edgelist,
    backend,
    quiet,
):
    """
    Validate the public inputs of ``compile_graph()``.

    Parameters
    ----------
    edgelist
        Edge table with required columns ``source`` and ``target``.
    backend
        The backend to compile to.
    quiet
        Whether informational notes should be suppressed.

    Raises
    ------
    KPNNError
        If any input is invalid.
    """
    if not isinstance(edgelist, pd.DataFrame):
        raise KPNNError("'edgelist' must be a pandas DataFrame.")

    required_columns = {"source", "target"}
    missing_columns = required_columns.difference(edgelist.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise KPNNError(
            "'edgelist' must contain the columns 'source' and "
            f"'target'. Missing: {missing_str}."
        )

    supported_backends = {"feedforward", "recurrent", "graph"}

    if backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise KPNNError(
            f"Unsupported backend '{backend}'. "
            f"Expected one of: {supported}."
        )

    if not isinstance(quiet, bool):
        raise KPNNError(
            "'quiet' must be a boolean value (True or False)."
        )
