"""
Artifact definition for compiled KPNN models.

Why this file exists
--------------------
This file defines the structured object that is returned alongside a
compiled PyTorch model. The artifact exists to preserve the non-model
metadata needed after compilation, especially for interpretation and
other downstream package operations.

Role in the package
-------------------
This is an internal compilation-bridge module. It defines the data
structure that connects graph compilation to later stages such as model
customization and interpretation. It should contain the artifact schema
itself, not compilation logic, validation logic, or model execution
code.
"""

from dataclasses import dataclass

from ..graph.schema import EdgeGraph


@dataclass
class CompileArtifact:
    """
    Compilation artifact returned together with the compiled PyTorch model.

    Parameters
    ----------
    backend : str
        Backend used for compilation.
    graph : EdgeGraph
        Internal KPNN graph object used for compilation.
    execution_plan
        Compiled execution plan used to build the model.
    node_names_by_layer : dict[str, list[str]]
        Mapping from layer name to node names in that layer.
    feature_names : list[str]
        Names of the input features.
    """

    backend: str
    graph: EdgeGraph
    execution_plan: object
    node_names_by_layer: dict[str, list[str]]
    feature_names: list[str]
