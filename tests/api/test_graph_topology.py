import pandas as pd
import pytest

from edge2torch import GraphTopology, compile_graph, graph_topology
from edge2torch.compile.artifact import CompileArtifact
from edge2torch.utils.errors import Edge2TorchError


def _compile_feedforward_artifact():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "gene_2"],
            "target": ["pathway_1", "pathway_1"],
        }
    )
    _, artifact = compile_graph(edgelist, backend="feedforward")
    return artifact


def _compile_state_update_artifact():
    edgelist = pd.DataFrame(
        {
            "source": ["input_1", "node_a", "node_b", "node_b"],
            "target": ["node_a", "node_b", "node_a", "output_1"],
        }
    )
    _, artifact = compile_graph(edgelist, backend="state_update")
    return artifact


def test_graph_topology_feedforward_matches_artifact_metadata():
    artifact = _compile_feedforward_artifact()

    topology = graph_topology(artifact)

    assert isinstance(topology, GraphTopology)
    assert topology.backend == "feedforward"
    assert topology.is_feedforward is True
    assert topology.is_state_update is False
    assert topology.feature_names == tuple(artifact.feature_names)
    assert topology.input_nodes == tuple(artifact.input_nodes)
    assert topology.output_nodes == tuple(artifact.output_nodes)
    assert topology.hidden_nodes == tuple(artifact.hidden_nodes)
    assert topology.interpretation_sites == {
        site_id: tuple(nodes)
        for site_id, nodes in artifact.interpretation_sites.items()
    }
    assert all(site_id.startswith("layer_") for site_id in topology.site_ids)


def test_graph_topology_state_update_matches_artifact_metadata():
    artifact = _compile_state_update_artifact()

    topology = graph_topology(artifact)

    assert topology.backend == "state_update"
    assert topology.is_feedforward is False
    assert topology.is_state_update is True
    assert topology.feature_names == tuple(artifact.feature_names)
    assert topology.interpretation_sites == {
        site_id: tuple(nodes)
        for site_id, nodes in artifact.interpretation_sites.items()
    }
    assert all(site_id.startswith("step_") for site_id in topology.site_ids)


def test_graph_topology_site_ids_are_numeric_sorted():
    artifact = _compile_state_update_artifact()
    artifact.interpretation_sites["step_10"] = ["node_a"]

    topology = graph_topology(artifact)

    step_2_index = topology.site_ids.index("step_2")
    step_10_index = topology.site_ids.index("step_10")
    assert step_2_index < step_10_index


def test_graph_topology_is_immutable():
    topology = graph_topology(_compile_feedforward_artifact())

    with pytest.raises(AttributeError):
        topology.backend = "state_update"  # type: ignore[misc]


def test_graph_topology_copies_interpretation_site_node_lists():
    artifact = _compile_feedforward_artifact()
    first_site_id = next(iter(artifact.interpretation_sites))
    original_nodes = tuple(artifact.interpretation_sites[first_site_id])

    topology = graph_topology(artifact)
    artifact.interpretation_sites[first_site_id].append("mutated_node")

    assert topology.interpretation_sites[first_site_id] == original_nodes


def test_graph_topology_rejects_non_artifact():
    with pytest.raises(Edge2TorchError, match="must be a CompileArtifact"):
        graph_topology(object())  # type: ignore[arg-type]


def test_graph_topology_rejects_unsupported_backend():
    artifact = CompileArtifact(
        backend="feedforward",  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        execution_plan=object(),
        node_names_by_layer={},
        input_nodes=["input_1"],
        output_nodes=["output_1"],
        hidden_nodes=[],
        interpretation_sites={"layer_1": ["input_1"]},
        feature_names=["input_1"],
    )
    artifact.backend = "legacy_backend"  # type: ignore[misc]

    with pytest.raises(Edge2TorchError, match="Unsupported artifact backend"):
        graph_topology(artifact)


def test_graph_topology_rejects_invalid_interpretation_site_id():
    artifact = _compile_feedforward_artifact()
    artifact.interpretation_sites["invalid_site"] = ["node_a"]

    with pytest.raises(Edge2TorchError, match="Invalid interpretation site"):
        graph_topology(artifact)
