# edge2torch

[![CI](https://github.com/Thomas-Rauter/edge2torch/actions/workflows/ci.yml/badge.svg)](https://github.com/Thomas-Rauter/edge2torch/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Thomas-Rauter/edge2torch/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Thomas-Rauter/edge2torch)

Build PyTorch models from edge lists of named neural architecture nodes.

![Graphical abstract of edge2torch](figures/graphical_abstract.svg)

## Overview

`edge2torch` is an edge-list-to-PyTorch compiler for sparse neural network
architectures with named nodes.

Define a model architecture as an edge list, compile it into a minimally
opinionated PyTorch model, train it with standard PyTorch tools, and optionally
map model behavior back to the named nodes and features that defined the
architecture.

The package is designed for users who want to build sparse or structured neural
networks from a predefined graph rather than manually wiring PyTorch modules.
It is domain-agnostic: any setting where a neural architecture can be
represented as named edges can use the same graph-to-model abstraction.

Here, "graph" means the architecture specification, not necessarily a graph
neural network. Feedforward models, recurrent models, and graph neural networks
can all be represented by edge lists when their architecture is defined through
directed connections between named nodes.

A major application area is knowledge-primed neural networks (KPNNs), where
prior knowledge defines the model structure. In biology, for example, edge lists
may connect genes, transcription factors, pathways, kinases, or other biological
entities. The same approach can also apply in domains such as chemistry or other
fields with graph-structured prior knowledge.

`edge2torch` deliberately leaves training loops, losses, optimizers,
task-specific heads, and advanced customization to standard PyTorch.

## Core workflow

The package is built around four main steps:

1. Define a model architecture as an edge list with named `source` and `target`
   nodes.
2. Compile the edge list into a backend-specific PyTorch model with
   `compile_graph()`.
3. Align named input data features to the compiled model input nodes with
   `align_features_to_input_nodes()`.
4. Customize, train, and interpret the model with ordinary PyTorch,
   `customize_model()`, and `interpret_model()`.

## Main public API

The current public API is centered on:

- `compile_graph()`
- `align_features_to_input_nodes()`
- `customize_model()`
- `interpret_model()`

## Package philosophy

`edge2torch` is intentionally minimally opinionated.

It defines the structural semantics required to compile a graph into a neural
network backend, but it does not impose broader modeling choices such as:

- activation functions
- output heads
- dropout
- loss functions
- optimizers
- training loops

These remain part of the normal PyTorch workflow.

This keeps the package small in scope:

- `edge2torch` handles graph compilation
- PyTorch handles model training
- `edge2torch` maps trained models back to interpretable named entities

## Supported backends

`compile_graph()` currently supports:

- `feedforward`
- `recurrent`
- `graphnn`

These backends share the same edge-list input format but differ in how the graph
structure is translated into neural-network computation.

Feature attribution is available through Captum-based methods. Feedforward
models also support broad node-level attribution. Recurrent and graph neural
network backends can be compiled and trained, while node-level interpretation
for these backends is planned for a future release.

See the **Backends** page for details.

## Start here

If you are new to the package, start with:

- **Installation** for package setup and optional extras
- **Getting started** for a full end-to-end example
- **Feedforward skip edges** for how non-adjacent feedforward edges are handled
- **Backends** for backend semantics and current support
- **Interpretation** for attribution targets, methods, and backend support
- **API reference** for function-level documentation

## Current scope

The package currently focuses on:

- compiling graph-defined architectures into PyTorch models
- aligning named data features to compiled model input nodes
- optional post-compilation model customization
- feature-level and node-level interpretation
- feedforward, recurrent, and graphnn backend support

Interpretation support is currently most complete for the feedforward backend.

## License

See `LICENSE`.
