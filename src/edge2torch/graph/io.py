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

from typing import cast

import pandas as pd

from .schema import EdgeGraph


def edgelist_to_graph(edgelist: pd.DataFrame) -> EdgeGraph:
    """
    Convert a validated edgelist DataFrame into the internal EdgeGraph object.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with validated ``source`` and ``target`` columns.
        May also contain optional sparse ``initial_weight`` and
        ``constraint`` columns.

    Returns
    -------
    EdgeGraph
        Internal graph object with normalized source and target columns and,
        when provided, normalized edge-level metadata.
    """
    edge_columns = ["source", "target"]

    if "initial_weight" in edgelist.columns:
        edge_columns.append("initial_weight")

    if "constraint" in edgelist.columns:
        edge_columns.append("constraint")

    edges = cast(pd.DataFrame, edgelist[edge_columns].copy())

    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()

    if "initial_weight" in edges.columns:
        edges["initial_weight"] = pd.to_numeric(
            edges["initial_weight"],
            errors="coerce",
        )

    if "constraint" in edges.columns:
        edges["constraint"] = (
            edges["constraint"]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("unconstrained")
        )

    edges = edges.reset_index(drop=True)

    return EdgeGraph(edges=edges)
