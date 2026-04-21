import pandas as pd


class KPNNGraph:
    """
    Internal graph representation used throughout the KPNN package.

    Parameters
    ----------
    edges : pd.DataFrame
        Normalized edge table with columns 'source' and 'target'.
    """

    def __init__(self, edges: pd.DataFrame):
        self.edges = edges
        self.nodes = self._extract_nodes()

    def _extract_nodes(self) -> list[str]:
        """
        Extract unique node names from the edge table.

        Returns
        -------
        list[str]
            Sorted list of unique node names.
        """
        nodes = set(self.edges["source"]).union(set(self.edges["target"]))
        return sorted(nodes)
