"""
Internal graph schema for KPNN compilation.

Why this file exists
--------------------
This file defines the package's internal graph representation so that
later compilation steps can work with one normalized graph object rather
than repeatedly reasoning about raw input tables. Keeping the graph
schema separate makes the internal compilation pipeline easier to reason
about and provides a stable structural boundary between input conversion
and backend-specific execution planning.

Role in the package
-------------------
This is an internal graph-schema module. It defines the normalized graph
object used throughout compilation and related internal graph
processing. It should contain the graph representation itself, not
public API validation, input-format conversion, or backend-specific
compilation logic.
"""

import pandas as pd


class EdgeGraph:
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
