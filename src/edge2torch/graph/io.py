"""
Conversion from public edgelist input to the internal graph schema.

Why this file exists
--------------------
This file isolates the step that turns the user-facing edgelist
representation into the internal graph object used throughout
compilation. The separation keeps graph normalization close to the graph
schema boundary and avoids mixing public input conversion with later
validation or backend-specific compilation logic.

Role in the package
-------------------
This is an internal graph-conversion module. It defines how accepted
edgelist input is normalized into the package's internal graph
representation. It should contain graph-format conversion logic, not
public API validation, backend compilation, or model execution code.
"""

import pandas as pd

from .schema import EdgeGraph


def edgelist_to_graph(edgelist: pd.DataFrame) -> EdgeGraph:
    """
    Convert a validated edgelist DataFrame into the internal EdgeGraph object.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with validated 'source' and 'target' columns.

    Returns
    -------
    EdgeGraph
        Internal graph object with normalized source and target columns.
    """
    edges = edgelist.loc[:, ["source", "target"]].copy()

    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()

    edges = edges.reset_index(drop=True)

    return EdgeGraph(edges=edges)
