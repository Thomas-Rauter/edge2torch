"""
Integration tests on a real tabular classification task.

These tests use the Breast Cancer Wisconsin Diagnostic dataset from
scikit-learn to verify that all edge2torch backends can compile a sparse
architecture, train with ordinary PyTorch, and achieve reasonable held-out
performance.

The purpose is not to benchmark edge2torch against specialized classifiers.
The purpose is to catch broken backend behavior that still passes smaller unit
tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge2torch import (
    align_features_to_input_nodes,
    compile_graph,
    customize_model,
)

pytestmark = pytest.mark.integration

SEED = 42


def test_feedforward_backend_learns_real_tabular_task() -> None:
    """Test that the feedforward backend learns a real tabular task."""
    metrics = _run_real_tabular_backend_task(
        backend="feedforward",
        make_edgelist=_make_feedforward_architecture_edgelist,
        n_outputs=16,
    )

    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["test_f1"] > 0.85
    assert metrics["test_roc_auc"] > 0.90


def test_recurrent_backend_learns_real_tabular_task() -> None:
    """Test that the recurrent backend learns a real tabular task."""
    metrics = _run_real_tabular_backend_task(
        backend="recurrent",
        make_edgelist=_make_recurrent_architecture_edgelist,
        n_outputs=8,
    )

    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["test_f1"] > 0.85
    assert metrics["test_roc_auc"] > 0.90


def test_graphnn_backend_learns_real_tabular_task() -> None:
    """Test that the graphnn backend learns a real tabular task."""
    metrics = _run_real_tabular_backend_task(
        backend="graphnn",
        make_edgelist=_make_graphnn_architecture_edgelist,
        n_outputs=8,
    )

    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["test_f1"] > 0.85
    assert metrics["test_roc_auc"] > 0.90


def _run_real_tabular_backend_task(
    backend: str,
    make_edgelist: Callable[[list[str]], pd.DataFrame],
    n_outputs: int,
) -> dict[str, float]:
    """Compile, train, and evaluate one backend on a real tabular task."""
    _set_seed(SEED)

    x_train_df, x_test_df, y_train, y_test = _make_breast_cancer_data()

    edgelist = make_edgelist(list(x_train_df.columns))

    base_model, artifact = compile_graph(
        edgelist=edgelist,
        backend=backend,
        quiet=True,
    )

    x_train_shuffled = x_train_df.loc[:, list(reversed(x_train_df.columns))]
    x_test_shuffled = x_test_df.loc[:, list(reversed(x_test_df.columns))]

    x_train = align_features_to_input_nodes(
        data=x_train_shuffled,
        artifact=artifact,
    )
    x_test = align_features_to_input_nodes(
        data=x_test_shuffled,
        artifact=artifact,
    )

    model = customize_model(
        model=base_model,
        activation=nn.ReLU(),
        dropout=0.10,
        head=nn.Linear(n_outputs, 1),
    )

    initial_loss, final_loss = _train_binary_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
    )

    test_f1, test_roc_auc = _evaluate_binary_classifier(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
    }


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible integration tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_breast_cancer_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    torch.Tensor,
    torch.Tensor,
]:
    """Load, split, scale, and return named tabular data."""
    dataset = load_breast_cancer(as_frame=True)

    x = dataset.data.copy()
    y = dataset.target.astype(np.float32).to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=x_test.columns,
    )

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return x_train_scaled, x_test_scaled, y_train_tensor, y_test_tensor


def _feature_statistic(feature_name: str) -> str:
    """Return the statistic prefix for a breast-cancer feature name."""
    if feature_name.startswith("mean "):
        return "mean"
    if feature_name.startswith("worst "):
        return "worst"
    if feature_name.endswith(" error"):
        return "error"

    raise ValueError(f"Unexpected feature name: {feature_name!r}")


def _feature_measurement(feature_name: str) -> str:
    """Return the measurement name without statistic-specific wording."""
    if feature_name.startswith("mean "):
        return feature_name.removeprefix("mean ")
    if feature_name.startswith("worst "):
        return feature_name.removeprefix("worst ")
    if feature_name.endswith(" error"):
        return feature_name.removesuffix(" error")

    raise ValueError(f"Unexpected feature name: {feature_name!r}")


def _make_feedforward_architecture_edgelist(
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a sparse feedforward architecture for the tabular task."""
    rng = np.random.default_rng(SEED)

    n_basis_outputs = 16
    basis_nodes = [
        f"classification_basis_{idx:02d}" for idx in range(n_basis_outputs)
    ]

    stat_nodes = {
        stat: f"statistic__{stat}"
        for stat in sorted({_feature_statistic(name) for name in feature_names})
    }
    measurement_nodes = {
        measurement: f"measurement__{measurement.replace(' ', '_')}"
        for measurement in sorted(
            {_feature_measurement(name) for name in feature_names}
        )
    }

    edges: list[tuple[str, str]] = []

    for feature_name in feature_names:
        stat = _feature_statistic(feature_name)
        measurement = _feature_measurement(feature_name)

        edges.append((feature_name, stat_nodes[stat]))
        edges.append((feature_name, measurement_nodes[measurement]))

    intermediate_nodes = list(stat_nodes.values()) + list(
        measurement_nodes.values()
    )

    for node in intermediate_nodes:
        targets = rng.choice(
            basis_nodes,
            size=4,
            replace=False,
        )
        edges.extend((node, str(target)) for target in targets)

    for feature_name in feature_names:
        if rng.random() < 0.35:
            target = rng.choice(basis_nodes)
            edges.append((feature_name, str(target)))

    return _make_edgelist(edges)


def _make_recurrent_architecture_edgelist(
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a cyclic sparse architecture for the recurrent backend."""
    rng = np.random.default_rng(SEED)

    n_state_nodes = 18
    n_basis_outputs = 8

    state_nodes = [f"state_{idx:02d}" for idx in range(n_state_nodes)]
    basis_nodes = [
        f"classification_basis_{idx:02d}" for idx in range(n_basis_outputs)
    ]

    statistic_groups = {
        "mean": [],
        "error": [],
        "worst": [],
    }
    measurement_groups: dict[str, list[str]] = {}

    for feature_name in feature_names:
        statistic_groups[_feature_statistic(feature_name)].append(feature_name)

        measurement = _feature_measurement(feature_name)
        measurement_groups.setdefault(measurement, []).append(feature_name)

    edges: list[tuple[str, str]] = []

    for idx, feature_name in enumerate(feature_names):
        primary_state = state_nodes[idx % n_state_nodes]
        secondary_state = state_nodes[(idx * 5 + 3) % n_state_nodes]

        edges.append((feature_name, primary_state))
        edges.append((feature_name, secondary_state))

    statistic_to_state = {
        "mean": state_nodes[0:6],
        "error": state_nodes[6:12],
        "worst": state_nodes[12:18],
    }

    for statistic, grouped_features in statistic_groups.items():
        target_states = statistic_to_state[statistic]
        for feature_name in grouped_features:
            target = rng.choice(target_states)
            edges.append((feature_name, str(target)))

    for idx, grouped_features in enumerate(measurement_groups.values()):
        target = state_nodes[idx % n_state_nodes]
        for feature_name in grouped_features:
            edges.append((feature_name, target))

    for idx, source in enumerate(state_nodes):
        edges.append((source, state_nodes[(idx + 1) % n_state_nodes]))
        edges.append((source, state_nodes[(idx - 3) % n_state_nodes]))

    for source in state_nodes:
        candidate_targets = [node for node in state_nodes if node != source]
        targets = rng.choice(candidate_targets, size=3, replace=False)
        edges.extend((source, str(target)) for target in targets)

    for source in state_nodes:
        targets = rng.choice(basis_nodes, size=3, replace=False)
        edges.extend((source, str(target)) for target in targets)

    return _make_edgelist(edges)


def _make_graphnn_architecture_edgelist(
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a graph-oriented sparse architecture for the graphnn backend."""
    rng = np.random.default_rng(SEED)

    n_state_nodes = 20
    n_basis_outputs = 8

    state_nodes = [f"graph_state_{idx:02d}" for idx in range(n_state_nodes)]
    basis_nodes = [
        f"classification_basis_{idx:02d}" for idx in range(n_basis_outputs)
    ]

    statistic_nodes = {
        "mean": state_nodes[0:6],
        "error": state_nodes[6:12],
        "worst": state_nodes[12:18],
    }

    measurements = sorted(
        {_feature_measurement(name) for name in feature_names}
    )
    measurement_to_state = {
        measurement: state_nodes[idx % n_state_nodes]
        for idx, measurement in enumerate(measurements)
    }

    edges: list[tuple[str, str]] = []

    for feature_name in feature_names:
        statistic = _feature_statistic(feature_name)
        measurement = _feature_measurement(feature_name)

        statistic_target = rng.choice(statistic_nodes[statistic])
        measurement_target = measurement_to_state[measurement]
        random_target = rng.choice(state_nodes)

        edges.append((feature_name, str(statistic_target)))
        edges.append((feature_name, measurement_target))
        edges.append((feature_name, str(random_target)))

    for idx, source in enumerate(state_nodes):
        edges.append((source, state_nodes[(idx + 1) % n_state_nodes]))
        edges.append((source, state_nodes[(idx - 1) % n_state_nodes]))
        edges.append((source, state_nodes[(idx + 5) % n_state_nodes]))
        edges.append((source, state_nodes[(idx - 7) % n_state_nodes]))

    for source in state_nodes:
        candidate_targets = [node for node in state_nodes if node != source]
        targets = rng.choice(candidate_targets, size=3, replace=False)
        edges.extend((source, str(target)) for target in targets)

    for source in state_nodes:
        targets = rng.choice(basis_nodes, size=3, replace=False)
        edges.extend((source, str(target)) for target in targets)

    return _make_edgelist(edges)


def _make_edgelist(edges: list[tuple[str, str]]) -> pd.DataFrame:
    """Create a clean edgelist DataFrame from edge tuples."""
    edgelist = pd.DataFrame(edges, columns=["source", "target"])
    return edgelist.drop_duplicates().reset_index(drop=True)


def _train_binary_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
) -> tuple[float, float]:
    """Train an ordinary PyTorch binary classifier and return losses."""
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    initial_loss = _compute_loss(
        model=model,
        x=x_train,
        y=y_train,
        loss_fn=loss_fn,
    )

    for _epoch in range(n_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    final_loss = _compute_loss(
        model=model,
        x=x_train,
        y=y_train,
        loss_fn=loss_fn,
    )

    return initial_loss, final_loss


def _compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    """Compute loss with the model in evaluation mode."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)

    model.train(was_training)

    return float(loss.detach())


def _evaluate_binary_classifier(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate F1 and ROC-AUC on the held-out test set."""
    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        probabilities = torch.sigmoid(logits)

    y_true = y_test.detach().cpu().numpy().reshape(-1)
    y_prob = probabilities.detach().cpu().numpy().reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    test_f1 = f1_score(y_true, y_pred)
    test_roc_auc = roc_auc_score(y_true, y_prob)

    return test_f1, test_roc_auc
