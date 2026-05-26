# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.

## [Unreleased]

### Added

### Changed

### Fixed

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
