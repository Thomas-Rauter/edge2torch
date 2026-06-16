# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.

## [Unreleased]

### Added

### Changed

### Fixed

## [0.2.0] - 2026-06-15

### Added

- Added unified node-level interpretation for the `recurrent` and `graphnn`
  backends.
- Added `level`, `nodes`, and `site_aggregation` parameters to
  `interpret_model()` for node interpretation.
- Added interpretation-site metadata to `CompileArtifact` (`input_nodes`,
  `output_nodes`, `hidden_nodes`, `interpretation_sites`).
- Added architecture-fixture tests for node interpretation across backends.
- Added recurrent and graphnn example notebooks for cyclic-graph end-to-end
  workflows.

### Changed

- **Breaking:** `interpret_model(..., target="nodes")` now returns a summary
  `pandas.DataFrame` by default (`level="summary"`, `nodes="hidden"`).
- **Breaking:** Per-site node tables now require `level="sites"`.
- Recurrent and graphnn models now unroll state updates into step modules for
  interpretation-site access.

### Fixed

- Fixed notebook JSON metadata required by `mkdocs build --strict`.
- Fixed graphnn example topology so the informative path reaches
  `readout_1` via `module_a`.

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
