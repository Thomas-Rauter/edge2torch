import pandas as pd
import pytest

from edge2torch.compile.interpretation_metadata import (
    build_feedforward_interpretation_sites,
    build_state_update_interpretation_sites,
    collect_feedforward_node_names,
    compute_hidden_nodes,
)
from edge2torch.compile_graph import compile_graph
from edge2torch.utils.constants import PSEUDO_NODE_PREFIX
from edge2torch.utils.errors import Edge2TorchError


def test_compute_hidden_nodes_excludes_inputs_outputs_and_pseudo_nodes():
    hidden_nodes = compute_hidden_nodes(
        node_names=[
            "gene_a",
            "pathway_1",
            f"{PSEUDO_NODE_PREFIX}gene_a__output__layer_1",
            "output_1",
        ],
        input_nodes=["gene_a"],
        output_nodes=["output_1"],
    )

    assert hidden_nodes == ["pathway_1"]


def test_build_feedforward_interpretation_sites_skips_input_layer():
    sites = build_feedforward_interpretation_sites(
        {
            "layer_0": ["gene_a", "gene_b"],
            "layer_1": ["pathway_1"],
            "layer_2": ["output_1"],
        }
    )

    assert sites == {
        "layer_1": ["pathway_1"],
        "layer_2": ["output_1"],
    }


def test_collect_feedforward_node_names_returns_sorted_unique_nodes():
    node_names = collect_feedforward_node_names(
        {
            "layer_0": ["gene_b", "gene_a"],
            "layer_1": ["pathway_1", "gene_a"],
        }
    )

    assert node_names == ["gene_a", "gene_b", "pathway_1"]


def test_build_state_update_interpretation_sites_uses_step_keys():
    sites = build_state_update_interpretation_sites(
        node_names=["gene_a", "hidden_1", "output_1"],
        steps=3,
    )

    assert list(sites.keys()) == ["step_1", "step_2", "step_3"]
    assert sites["step_1"] == ["gene_a", "hidden_1", "output_1"]
    assert sites["step_3"] == sites["step_1"]


def test_build_state_update_interpretation_sites_rejects_non_positive_steps():
    with pytest.raises(Edge2TorchError, match="positive integer"):
        build_state_update_interpretation_sites(
            node_names=["gene_a"],
            steps=0,
        )


@pytest.mark.parametrize("backend", ["state_update"])
def test_compile_graph_builds_state_update_interpretation_metadata(
    backend: str,
):
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "hidden_1", "hidden_1"],
            "target": ["hidden_1", "hidden_1", "output_1"],
        }
    )

    _, artifact = compile_graph(
        edgelist=edgelist,
        backend=backend,
        quiet=True,
        steps=4,
    )

    assert artifact.input_nodes == ["gene_1"]
    assert artifact.output_nodes == ["output_1"]
    assert artifact.hidden_nodes == ["hidden_1"]
    assert list(artifact.interpretation_sites.keys()) == [
        "step_1",
        "step_2",
        "step_3",
        "step_4",
    ]
    assert artifact.interpretation_sites["step_2"] == sorted(
        ["gene_1", "hidden_1", "output_1"]
    )


def test_compile_graph_feedforward_skip_edges_hide_pseudo_nodes():
    edgelist = pd.DataFrame(
        {
            "source": ["gene_1", "pathway_1", "pathway_2", "gene_1"],
            "target": ["pathway_1", "pathway_2", "output_1", "output_1"],
        }
    )

    _, artifact = compile_graph(edgelist, quiet=True)

    assert all(
        not node_name.startswith(PSEUDO_NODE_PREFIX)
        for node_name in artifact.hidden_nodes
    )

    pseudo_nodes = [
        node_name
        for site_nodes in artifact.interpretation_sites.values()
        for node_name in site_nodes
        if node_name.startswith(PSEUDO_NODE_PREFIX)
    ]

    assert pseudo_nodes

    assert not any(
        node_name.startswith(PSEUDO_NODE_PREFIX)
        for node_name in artifact.hidden_nodes
    )
