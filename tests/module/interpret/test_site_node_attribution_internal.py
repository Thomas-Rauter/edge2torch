import pandas as pd
import pytest
import torch
from torch import nn

from edge2torch.compile.artifact import CompileArtifact
from edge2torch.graph.schema import EdgeGraph
from edge2torch.interpret.site_node_attribution import (
    _aggregate_state_update_site_attributions,
    _build_site_interpreter,
    _build_summary_attribution,
    _filter_site_attributions,
    _is_visible_domain_node,
    _merge_feedforward_site_attributions,
    _should_include_node,
    _site_sort_key,
    _validate_node_attributions,
    run_site_node_attribution,
)
from edge2torch.utils.constants import INTERNAL_NODE_PREFIX, PSEUDO_NODE_PREFIX
from edge2torch.utils.errors import Edge2TorchError


class _TinyModel(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


class _Artifact:
    def __init__(self, *, backend="feedforward", hidden_nodes=None):
        self.backend = backend
        self.hidden_nodes = (
            ["hidden_1"] if hidden_nodes is None else hidden_nodes
        )
        self.input_nodes = ["gene_a"]
        self.output_nodes = ["output_1"]


def test_build_site_interpreter_rejects_unknown_method():
    model = _TinyModel()
    site_module = nn.Linear(2, 1)

    with pytest.raises(
        Edge2TorchError,
        match="not supported for target='nodes'",
    ):
        _build_site_interpreter(
            method="not_a_method",
            model=model,
            site_module=site_module,
            constructor_kwargs={},
        )


def test_validate_node_attributions_rejects_non_2d_tensor():
    attributions = torch.randn(2, 3, 1)

    with pytest.raises(
        Edge2TorchError,
        match="shape \\(n_examples, n_nodes\\)",
    ):
        _validate_node_attributions(
            attributions=attributions,
            site_id="layer_1",
            sample_names=["sample_1", "sample_2"],
            node_names=["node_1", "node_2", "node_3"],
        )


def test_validate_node_attributions_accepts_matching_shape():
    attributions = torch.randn(2, 2)

    _validate_node_attributions(
        attributions=attributions,
        site_id="layer_1",
        sample_names=["sample_1", "sample_2"],
        node_names=["node_1", "node_2"],
    )


def test_is_visible_domain_node_rejects_internal_and_pseudo_nodes():
    assert _is_visible_domain_node(f"{INTERNAL_NODE_PREFIX}skip") is False
    assert _is_visible_domain_node(f"{PSEUDO_NODE_PREFIX}skip") is False
    assert _is_visible_domain_node("gene_a") is True


def test_site_sort_key_sorts_layers_before_steps():
    assert _site_sort_key("layer_2") < _site_sort_key("step_1")


def test_should_include_node_respects_hidden_filter():
    artifact = _Artifact(hidden_nodes=["hidden_1"])

    assert _should_include_node("hidden_1", artifact, "hidden") is True
    assert _should_include_node("gene_a", artifact, "hidden") is False
    assert _should_include_node("gene_a", artifact, "all") is True
    assert _should_include_node("gene_a", artifact, "non_input") is False
    assert _should_include_node("output_1", artifact, "non_input") is True


def test_filter_site_attributions_returns_selected_columns():
    attributions = torch.tensor([[1.0, 2.0, 3.0]])
    artifact = _Artifact(hidden_nodes=["hidden_1"])

    node_names, filtered = _filter_site_attributions(
        attributions=attributions,
        node_names=["gene_a", "hidden_1", "output_1"],
        artifact=artifact,
        nodes="hidden",
    )

    assert node_names == ["hidden_1"]
    assert torch.equal(filtered, torch.tensor([[2.0]]))


def test_merge_feedforward_site_attributions_concatenates_disjoint_columns():
    site_results = {
        "layer_1": pd.DataFrame(
            [[1.0, 2.0]],
            index=["sample_1"],
            columns=["hidden_1", "hidden_2"],
        ),
        "layer_2": pd.DataFrame(
            [[3.0]],
            index=["sample_1"],
            columns=["output_1"],
        ),
    }

    summary = _merge_feedforward_site_attributions(site_results)

    assert list(summary.columns) == ["hidden_1", "hidden_2", "output_1"]
    assert summary.loc["sample_1", "hidden_2"] == 2.0


def test_aggregate_state_update_site_attributions_uses_last_step():
    site_results = {
        "step_1": pd.DataFrame([[1.0]], index=["s1"], columns=["hidden_1"]),
        "step_2": pd.DataFrame([[4.0]], index=["s1"], columns=["hidden_1"]),
    }

    summary = _aggregate_state_update_site_attributions(
        site_results=site_results,
        site_aggregation="last",
    )

    assert summary.loc["s1", "hidden_1"] == 4.0


def test_aggregate_state_update_site_attributions_uses_max_abs():
    site_results = {
        "step_1": pd.DataFrame([[-5.0]], index=["s1"], columns=["hidden_1"]),
        "step_2": pd.DataFrame([[2.0]], index=["s1"], columns=["hidden_1"]),
    }

    summary = _aggregate_state_update_site_attributions(
        site_results=site_results,
        site_aggregation="max_abs",
    )

    assert summary.loc["s1", "hidden_1"] == -5.0


def test_aggregate_state_update_site_attributions_uses_mean_abs():
    site_results = {
        "step_1": pd.DataFrame([[-4.0]], index=["s1"], columns=["hidden_1"]),
        "step_2": pd.DataFrame([[2.0]], index=["s1"], columns=["hidden_1"]),
    }

    summary = _aggregate_state_update_site_attributions(
        site_results=site_results,
        site_aggregation="mean_abs",
    )

    assert summary.loc["s1", "hidden_1"] == pytest.approx(3.0)


def test_build_summary_attribution_rejects_empty_site_results():
    with pytest.raises(Edge2TorchError, match="empty site results"):
        _build_summary_attribution(
            artifact=_Artifact(backend="state_update"),
            site_results={},
            site_aggregation="max_abs",
        )


def test_build_summary_attribution_rejects_unsupported_backend():
    site_results = {
        "step_1": pd.DataFrame([[1.0]], index=["s1"], columns=["hidden_1"]),
    }

    with pytest.raises(Edge2TorchError, match="Unsupported backend"):
        _build_summary_attribution(
            artifact=_Artifact(backend="legacy"),
            site_results=site_results,
            site_aggregation="max_abs",
        )


def test_site_sort_key_rejects_malformed_ids():
    with pytest.raises(Edge2TorchError, match="Invalid interpretation site"):
        _site_sort_key("layer_x")

    with pytest.raises(Edge2TorchError, match="Invalid interpretation site"):
        _site_sort_key("step_x")

    with pytest.raises(Edge2TorchError, match="Invalid interpretation site"):
        _site_sort_key("not_a_site")


def test_should_include_node_rejects_unknown_filter():
    artifact = _Artifact()

    with pytest.raises(Edge2TorchError, match="Unsupported node filter"):
        _should_include_node(
            "hidden_1",
            artifact,
            "outputs",  # type: ignore[arg-type]
        )


def test_validate_node_attributions_rejects_sample_or_width_mismatch():
    with pytest.raises(Edge2TorchError, match="row count"):
        _validate_node_attributions(
            attributions=torch.randn(1, 2),
            site_id="layer_1",
            sample_names=["s1", "s2"],
            node_names=["n1", "n2"],
        )

    with pytest.raises(Edge2TorchError, match="width mismatch"):
        _validate_node_attributions(
            attributions=torch.randn(2, 1),
            site_id="layer_1",
            sample_names=["s1", "s2"],
            node_names=["n1", "n2"],
        )


def test_run_site_node_attribution_returns_summary_dataframe():
    class _FeedforwardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Linear(2, 1, bias=False)

        def forward(self, x):
            return self.block(x)

        def _edge2torch_get_interpretation_site(self, site_id: str):
            if site_id != "layer_1":
                raise ValueError(site_id)

            return self.block

    model = _FeedforwardModel()
    artifact = CompileArtifact(
        backend="feedforward",
        graph=EdgeGraph(
            edges=pd.DataFrame(
                {
                    "source": ["gene_a", "gene_b"],
                    "target": ["pathway_1", "pathway_1"],
                }
            )
        ),
        execution_plan=object(),
        node_names_by_layer={
            "layer_0": ["gene_a", "gene_b"],
            "layer_1": ["pathway_1"],
        },
        input_nodes=["gene_a", "gene_b"],
        output_nodes=["pathway_1"],
        hidden_nodes=["pathway_1"],
        interpretation_sites={"layer_1": ["pathway_1"]},
        feature_names=["gene_a", "gene_b"],
    )

    result = run_site_node_attribution(
        model=model,
        artifact=artifact,
        inputs=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        sample_names=["s1", "s2"],
        method="LayerActivation",
        constructor_kwargs={},
        attribute_kwargs={},
        level="summary",
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["pathway_1"]
    assert result.shape == (2, 1)
