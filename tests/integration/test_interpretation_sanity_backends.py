"""
Integration tests that train synthetic signal/decoy graphs and verify
interpretation separates causal subgraph nodes and features from decoys.

Each backend uses a parallel signal tower (labels depend only on signal
inputs) and a decoy tower that is wired but not on any path to the loss.
After a learnability gate on held-out data, node and feature attributions
must satisfy:

1. Every signal item ranks above every decoy item (min(signal) > max(decoy)).
2. The best decoy is below 10% of the weakest signal
   (max(decoy) < 0.1 * min(signal)).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge2torch import (
    align_features_to_input_nodes,
    compile_graph,
    customize_model,
    interpret_model,
)

pytestmark = pytest.mark.integration

BackendName = Literal["feedforward", "recurrent", "graphnn"]

SEED = 7
N_TRAIN = 512
N_VAL = 256
N_EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
MIN_VAL_ROC_AUC = 0.95
DECOY_TO_SIGNAL_RATIO = 0.1

SIGNAL_FEATURES = [f"sig_in_{idx}" for idx in range(4)]
DECOY_FEATURES = [f"dec_in_{idx}" for idx in range(4)]
SIGNAL_HIDDEN = [f"sig_h_{idx}" for idx in range(5)]
DECOY_HIDDEN = [f"dec_h_{idx}" for idx in range(4)]
OUTPUT_NODE = "prediction"
DECOY_OUTPUT_NODE = "decoy_readout"

SIGNAL_FEATURE_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float64)


@dataclass(frozen=True)
class SignalDecoyGraph:
    """Synthetic graph metadata for interpretation sanity tests."""

    edgelist: pd.DataFrame
    signal_features: tuple[str, ...]
    decoy_features: tuple[str, ...]
    signal_hidden: tuple[str, ...]
    decoy_hidden: tuple[str, ...]


class _PredictionOutputModel(nn.Module):
    """Expose only the signal output while preserving interpretation sites."""

    def __init__(self, inner: nn.Module, output_index: int) -> None:
        super().__init__()
        self.inner = inner
        self.output_index = output_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.inner(x)
        return outputs[:, self.output_index : self.output_index + 1]

    def _edge2torch_list_interpretation_site_ids(self) -> list[str]:
        return self.inner._edge2torch_list_interpretation_site_ids()

    def _edge2torch_get_interpretation_site(self, site_id: str) -> nn.Module:
        return self.inner._edge2torch_get_interpretation_site(site_id)


class _InterpretableTaskModel(nn.Module):
    """Task head on the signal output with interpretation-site delegation."""

    def __init__(self, prediction_model: _PredictionOutputModel) -> None:
        super().__init__()
        self.prediction_model = prediction_model
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.prediction_model(x))

    def _edge2torch_list_interpretation_site_ids(self) -> list[str]:
        return self.prediction_model._edge2torch_list_interpretation_site_ids()

    def _edge2torch_get_interpretation_site(self, site_id: str) -> nn.Module:
        return self.prediction_model._edge2torch_get_interpretation_site(
            site_id
        )


@pytest.mark.parametrize("backend", ["feedforward", "recurrent", "graphnn"])
def test_learned_attributions_separate_signal_from_decoy(
    backend: BackendName,
) -> None:
    """Train on a signal/decoy graph and verify attributions."""
    graph = _build_signal_decoy_graph(backend)
    _set_seed(SEED)

    x_train_df, x_val_df, y_train, y_val = _make_synthetic_classification_data(
        signal_features=graph.signal_features,
        decoy_features=graph.decoy_features,
    )

    compile_kwargs: dict = {
        "edgelist": graph.edgelist,
        "backend": backend,
        "quiet": True,
    }
    if backend in {"recurrent", "graphnn"}:
        compile_kwargs["steps"] = 5

    base_model, artifact = compile_graph(**compile_kwargs)

    _assert_graph_metadata_matches(graph, artifact)

    x_train = align_features_to_input_nodes(
        data=x_train_df,
        artifact=artifact,
    )
    x_val = align_features_to_input_nodes(
        data=x_val_df,
        artifact=artifact,
    )

    prediction_index = artifact.output_nodes.index(OUTPUT_NODE)

    customized_model = customize_model(
        model=base_model,
        dropout=0.05,
    )
    prediction_model = _PredictionOutputModel(
        customized_model, prediction_index
    )
    model = _InterpretableTaskModel(prediction_model)

    _train_binary_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
    )

    val_roc_auc = _evaluate_roc_auc(
        model=model,
        x_val=x_val,
        y_val=y_val,
    )
    assert val_roc_auc >= MIN_VAL_ROC_AUC, (
        f"Model did not learn synthetic task (val ROC-AUC={val_roc_auc:.3f})"
    )

    node_summary = interpret_model(
        model=model,
        artifact=artifact,
        data=x_val_df,
        target="nodes",
        method="LayerConductance",
        quiet=True,
        level="summary",
        nodes="hidden",
    )
    feature_attributions = interpret_model(
        model=model,
        artifact=artifact,
        data=x_val_df,
        target="features",
        method="IntegratedGradients",
        quiet=True,
    )

    _assert_signal_decoy_separation(
        scores=_median_abs_scores(node_summary, graph.signal_hidden),
        decoy_scores=_median_abs_scores(node_summary, graph.decoy_hidden),
        label="hidden nodes",
    )
    _assert_signal_decoy_separation(
        scores=_median_abs_scores(feature_attributions, graph.signal_features),
        decoy_scores=_median_abs_scores(
            feature_attributions, graph.decoy_features
        ),
        label="input features",
    )


def _build_signal_decoy_graph(backend: BackendName) -> SignalDecoyGraph:
    """Return a backend-specific signal/decoy edgelist and node groupings."""
    if backend == "feedforward":
        return _build_feedforward_signal_decoy_graph()
    if backend == "recurrent":
        return _build_recurrent_signal_decoy_graph()
    return _build_graphnn_signal_decoy_graph()


def _build_feedforward_signal_decoy_graph() -> SignalDecoyGraph:
    edges: list[tuple[str, str]] = []

    for idx, feature_name in enumerate(SIGNAL_FEATURES):
        edges.append((feature_name, SIGNAL_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("sig_h_0", "sig_h_2"),
            ("sig_h_0", "sig_h_3"),
            ("sig_h_1", "sig_h_2"),
            ("sig_h_1", "sig_h_3"),
            ("sig_h_2", "sig_h_4"),
            ("sig_h_3", "sig_h_4"),
            ("sig_h_4", OUTPUT_NODE),
        ]
    )

    for idx, feature_name in enumerate(DECOY_FEATURES):
        edges.append((feature_name, DECOY_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("dec_h_0", "dec_h_2"),
            ("dec_h_1", "dec_h_2"),
            ("dec_h_1", "dec_h_3"),
            ("dec_h_2", "dec_h_3"),
            ("dec_h_3", DECOY_OUTPUT_NODE),
        ]
    )

    return SignalDecoyGraph(
        edgelist=_make_edgelist(edges),
        signal_features=tuple(SIGNAL_FEATURES),
        decoy_features=tuple(DECOY_FEATURES),
        signal_hidden=tuple(SIGNAL_HIDDEN),
        decoy_hidden=tuple(DECOY_HIDDEN),
    )


def _build_recurrent_signal_decoy_graph() -> SignalDecoyGraph:
    edges: list[tuple[str, str]] = []

    for idx, feature_name in enumerate(SIGNAL_FEATURES):
        edges.append((feature_name, SIGNAL_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("sig_h_0", "sig_h_1"),
            ("sig_h_0", "sig_h_2"),
            ("sig_h_1", "sig_h_2"),
            ("sig_h_2", "sig_h_1"),
            ("sig_h_1", "sig_h_3"),
            ("sig_h_2", "sig_h_3"),
            ("sig_h_3", "sig_h_4"),
            ("sig_h_4", OUTPUT_NODE),
        ]
    )

    for idx, feature_name in enumerate(DECOY_FEATURES):
        edges.append((feature_name, DECOY_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("dec_h_0", "dec_h_1"),
            ("dec_h_1", "dec_h_2"),
            ("dec_h_2", "dec_h_3"),
            ("dec_h_3", "dec_h_0"),
            ("dec_h_0", "dec_h_2"),
            ("dec_h_1", "dec_h_3"),
        ]
    )

    return SignalDecoyGraph(
        edgelist=_make_edgelist(edges),
        signal_features=tuple(SIGNAL_FEATURES),
        decoy_features=tuple(DECOY_FEATURES),
        signal_hidden=tuple(SIGNAL_HIDDEN),
        decoy_hidden=tuple(DECOY_HIDDEN),
    )


def _build_graphnn_signal_decoy_graph() -> SignalDecoyGraph:
    edges: list[tuple[str, str]] = []

    for idx, feature_name in enumerate(SIGNAL_FEATURES):
        edges.append((feature_name, SIGNAL_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("sig_h_0", "sig_h_1"),
            ("sig_h_0", "sig_h_3"),
            ("sig_h_1", "sig_h_2"),
            ("sig_h_2", "sig_h_1"),
            ("sig_h_1", "sig_h_4"),
            ("sig_h_2", "sig_h_4"),
            ("sig_h_3", "sig_h_4"),
            ("sig_h_4", OUTPUT_NODE),
        ]
    )

    for idx, feature_name in enumerate(DECOY_FEATURES):
        edges.append((feature_name, DECOY_HIDDEN[idx % 2]))

    edges.extend(
        [
            ("dec_h_0", "dec_h_1"),
            ("dec_h_1", "dec_h_2"),
            ("dec_h_2", "dec_h_3"),
            ("dec_h_3", "dec_h_0"),
            ("dec_h_0", "dec_h_3"),
            ("dec_h_1", "dec_h_3"),
        ]
    )

    return SignalDecoyGraph(
        edgelist=_make_edgelist(edges),
        signal_features=tuple(SIGNAL_FEATURES),
        decoy_features=tuple(DECOY_FEATURES),
        signal_hidden=tuple(SIGNAL_HIDDEN),
        decoy_hidden=tuple(DECOY_HIDDEN),
    )


def _make_edgelist(edges: list[tuple[str, str]]) -> pd.DataFrame:
    edgelist = pd.DataFrame(edges, columns=["source", "target"])
    return edgelist.drop_duplicates().reset_index(drop=True)


def _make_synthetic_classification_data(
    *,
    signal_features: tuple[str, ...],
    decoy_features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Build splits where labels depend only on signal inputs."""
    rng = np.random.default_rng(SEED)

    feature_names = list(signal_features) + list(decoy_features)
    n_features = len(feature_names)

    x = rng.normal(loc=0.0, scale=1.0, size=(N_TRAIN + N_VAL, n_features))
    signal_values = x[:, : len(signal_features)]
    linear_score = (
        signal_values @ SIGNAL_FEATURE_WEIGHTS[: len(signal_features)]
    )
    y = (linear_score > 0.0).astype(np.float32).reshape(-1, 1)

    x_df = pd.DataFrame(x, columns=feature_names)
    y_series = pd.Series(y.reshape(-1))

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_df,
        y_series,
        test_size=N_VAL,
        random_state=SEED,
        stratify=y_series,
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    )
    y_val_tensor = torch.tensor(y_val.to_numpy(dtype=np.float32).reshape(-1, 1))

    return x_train_df, x_val_df, y_train_tensor, y_val_tensor


def _assert_graph_metadata_matches(graph: SignalDecoyGraph, artifact) -> None:
    assert set(graph.signal_features).issubset(artifact.feature_names)
    assert set(graph.decoy_features).issubset(artifact.feature_names)
    assert set(graph.signal_hidden).issubset(artifact.hidden_nodes)
    assert set(graph.decoy_hidden).issubset(artifact.hidden_nodes)
    assert OUTPUT_NODE in artifact.output_nodes


def _median_abs_scores(
    attribution_df: pd.DataFrame,
    names: tuple[str, ...],
) -> dict[str, float]:
    return {name: float(attribution_df[name].abs().median()) for name in names}


def _assert_signal_decoy_separation(
    *,
    scores: dict[str, float],
    decoy_scores: dict[str, float],
    label: str,
) -> None:
    signal_floor = min(scores.values())
    decoy_ceiling = max(decoy_scores.values())

    assert signal_floor > decoy_ceiling, (
        f"{label}: expected all signal scores above all decoy scores, "
        f"but min(signal)={signal_floor:.6f} and max(decoy)={decoy_ceiling:.6f}"
    )
    assert decoy_ceiling < DECOY_TO_SIGNAL_RATIO * signal_floor, (
        f"{label}: expected max(decoy) < {DECOY_TO_SIGNAL_RATIO:.0%} of "
        f"min(signal), but max(decoy)={decoy_ceiling:.6f} and "
        f"min(signal)={signal_floor:.6f}"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_binary_classifier(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> None:
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _epoch in range(N_EPOCHS):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()


def _evaluate_roc_auc(
    *,
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        logits = model(x_val)
        probabilities = torch.sigmoid(logits)

    y_true = y_val.detach().cpu().numpy().reshape(-1)
    y_prob = probabilities.detach().cpu().numpy().reshape(-1)

    return float(roc_auc_score(y_true, y_prob))
