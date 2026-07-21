# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [0.3.0] - 2026-07-17

### Removed

- **`backend="recurrent"` and `backend="graphnn"`.** The public API now has
  two backends only: `feedforward` and `state_update`.
- **`RecurrentEdgeModel` and `EdgeGraphNNModel`.** Cyclic graphs compile to
  `StateUpdateEdgeModel`.
- **`model.recurrent` and `model.message_passing`.** Use `model.state_linear`
  for the shared masked linear layer.

### Added

- **`backend="state_update"`** as the only topology-preserving compile path for
  cyclic graphs.
- **`StateUpdateEdgeModel`** as the compiled model class for `state_update`.
- **`model.state_linear`** as the canonical accessor for state-update weights.

### Changed

- **Docs and examples** now describe `feedforward` and `state_update` only. The
  cyclic-graph walkthrough is `docs/state-update-example.ipynb`.
- **Node interpretation on `state_update`** still uses `step_*` sites and the
  same `site_aggregation` options as before.

## [0.2.0] - 2026-06-26

### Added

- **Node interpretation on all backends.** You can now attribute predictions to
  named hidden nodes on `recurrent` and `graphnn` models, not only on
  `feedforward`.
- **Finer control over node attribution output** via new `interpret_model()`
  options:
  - `level="summary"` — one table per sample (default)
  - `level="sites"` — separate tables per interpretation site (`layer_*` on
    feedforward, `step_*` on recurrent and graphnn)
  - `nodes` — include hidden nodes only, or also outputs (`"non_input"`), or
    all visible nodes (`"all"`)
  - `site_aggregation` — on recurrent and graphnn, choose how step-wise scores
    are combined in the summary (`"max_abs"`, `"mean_abs"`, or `"last"`)
- **New example notebooks:** recurrent and graphnn end-to-end workflows
  (compile, train, interpret on cyclic graphs).
- **New docs page:** [Scope and limitations](https://thomas-rauter.github.io/edge2torch/0.2.0/scope/)
  — what each backend supports cleanly, with extra PyTorch work, or not at all.

### Changed

- **Breaking:** `interpret_model(..., target="nodes")` now returns a summary
  `pandas.DataFrame` by default. For the previous per-site dictionary of
  tables, pass `level="sites"`.

### Fixed

- GraphNN example notebook: corrected graph topology so the signal path
  reaches the readout node as intended.

## [0.1.0] - 2026-05-26

### Added

- Initial release of `edge2torch`.
- Added `compile_graph()` for compiling named edge lists into PyTorch models.
- Added support for the `feedforward`, `recurrent`, and `graphnn` backends.
- Added feature alignment with `align_features_to_input_nodes()`.
- Added model customization with `customize_model()`.
- Added Captum-based interpretation with `interpret_model()`, including
  feature-level attribution and feedforward node-level attribution.
- Added documentation, examples, and tests.
