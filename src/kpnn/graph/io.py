import pandas as pd

from .schema import KPNNGraph


def edgelist_to_graph(edgelist: pd.DataFrame) -> KPNNGraph:
    """
    Convert an edgelist DataFrame into the internal KPNNGraph object.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with columns 'source' and 'target'.

    Returns
    -------
    KPNNGraph
        Internal graph object with normalized edges.
    """
    edges = edgelist.loc[:, ["source", "target"]].copy()

    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()

    edges = edges.reset_index(drop=True)

    return KPNNGraph(edges=edges)
