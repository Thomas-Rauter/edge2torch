# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.

## [Unreleased]

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
- **New docs page:** [Scope and limitations](https://Thomas-Rauter.github.io/edge2torch/scope/)
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
