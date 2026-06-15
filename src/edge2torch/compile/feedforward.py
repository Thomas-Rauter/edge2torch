"""
Feedforward backend compilation.

Why this file exists
--------------------
This file isolates the logic for compiling a validated graph into the
feedforward backend. The separation keeps backend-specific compilation
out of the general compiler dispatch and makes the feedforward path easy
to understand and extend independently.

Role in the package
-------------------
This is an internal backend-compilation module. It takes a graph,
derives the feedforward execution plan, builds the corresponding PyTorch
model, and packages the result together with its artifact. It should
contain feedforward-specific compilation logic, not public API handling
or generic backend dispatch.
"""

from ..graph.schema import EdgeGraph
from ..nn.model import EdgeModel
from .artifact import CompileArtifact
from .execution_plan import build_feedforward_execution_plan
from .interpretation_metadata import (
    build_feedforward_interpretation_sites,
    collect_feedforward_node_names,
    compute_hidden_nodes,
)


def compile_feedforward(
    graph: EdgeGraph,
    bias: bool = True,
) -> tuple[EdgeModel, CompileArtifact]:
    """
    Compile a KPNN graph into a feedforward PyTorch model.

    Parameters
    ----------
    graph
        Internal KPNN graph object.
    bias
        Whether compiled masked linear layers include bias terms. If True,
        each target node has a learned node-level offset in addition to its
        graph-defined weighted inputs. If False, node updates are computed only
        from graph-defined weighted inputs. Disabling bias gives the graph
        structure stricter control over node activations.

    Returns
    -------
    tuple
        A tuple of (model, artifact).
    """
    # Decides the structure
    execution_plan = build_feedforward_execution_plan(graph)

    # Builds the actual PyTorch modules from that structure.
    model = EdgeModel(
        execution_plan=execution_plan,
        bias=bias,
    )

    # Stores the metadata needed later for alignment, interpretation, and
    # inspection.
    node_names_by_layer = execution_plan.node_names_by_layer
    input_nodes = list(execution_plan.input_node_names)
    output_nodes = list(execution_plan.output_node_names)

    artifact = CompileArtifact(
        backend="feedforward",
        graph=graph,
        execution_plan=execution_plan,
        node_names_by_layer=node_names_by_layer,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        hidden_nodes=compute_hidden_nodes(
            node_names=collect_feedforward_node_names(node_names_by_layer),
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        ),
        interpretation_sites=build_feedforward_interpretation_sites(
            node_names_by_layer
        ),
        feature_names=input_nodes,
    )

    return model, artifact
