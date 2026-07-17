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
    Compilation metadata returned together with the compiled PyTorch model.

    ``CompileArtifact`` is returned by ``compile_graph()`` and accepted by
    public helper functions such as ``align_features_to_input_nodes()`` and
    ``interpret_model()``. It is exported for user-facing type hints and
    workflow integration.

    The stable user-facing fields are ``backend`` and ``feature_names``.
    Other fields expose compilation internals for inspection, testing, and
    debugging. They may change across releases and should not be treated as
    part of the stable public API.

    Parameters
    ----------
    backend : str
        Backend used for compilation.
    graph : EdgeGraph
        Internal edge2torch graph object used for compilation. The graph
        contains the normalized edge table and may include optional edge-level
        metadata such as ``initial_weight`` and ``constraint``. This field is
        intended for inspection and debugging and may change across releases.
    execution_plan
        Compiled execution plan used to build the model. This field exposes
        backend-specific internals and may change across releases.
    node_names_by_layer : dict[str, list[str]]
        Mapping from layer name to node names in that layer. This field is
        populated for the feedforward backend and is primarily intended for
        inspection and feedforward-backend internals.
    input_nodes : list[str]
        Names of graph input nodes inferred as nodes with no incoming edges.
    output_nodes : list[str]
        Names of graph output nodes inferred as nodes with no outgoing edges.
    hidden_nodes : list[str]
        Names of hidden graph nodes excluding inputs, outputs, and compiler
        pseudo nodes.
    interpretation_sites : dict[str, list[str]]
        Mapping from interpretation site identifier to ordered node names for
        that site. Feedforward backends use ``layer_1``, ``layer_2``, and so on.
        State-update backends use ``step_1``, ``step_2``, and so on.
    feature_names : list[str]
        Names of the input features. This field defines the expected input
        column order for tensors passed to the compiled model.
    """

    backend: str
    graph: EdgeGraph
    execution_plan: object
    node_names_by_layer: dict[str, list[str]]
    input_nodes: list[str]
    output_nodes: list[str]
    hidden_nodes: list[str]
    interpretation_sites: dict[str, list[str]]
    feature_names: list[str]
