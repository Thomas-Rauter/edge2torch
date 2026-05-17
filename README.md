# edge2torch

[![CI](https://github.com/Thomas-Rauter/edge2torch/actions/workflows/ci.yml/badge.svg)](https://github.com/Thomas-Rauter/edge2torch/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Thomas-Rauter/edge2torch/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Thomas-Rauter/edge2torch)

Build sparse PyTorch neural networks from edge lists of named nodes,
with optional feature- and node-level attribution.

**Documentation:** https://Thomas-Rauter.github.io/edge2torch/

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

A major application area is knowledge-primed neural networks (KPNNs), where
prior knowledge defines the model structure. In biology, for example, edge lists
may connect genes, transcription factors, pathways, kinases, or other biological
entities. The same approach can also apply in domains such as chemistry or other
fields with graph-structured prior knowledge.

`edge2torch` deliberately leaves training loops, losses, optimizers,
task-specific heads, and advanced customization to standard PyTorch.

The package is built around four main steps:

1. Define a model architecture as an edge list with named `source` and `target`
   nodes.
2. Compile the edge list into a backend-specific PyTorch model with
   `compile_graph()`.
3. Align named input data features to the compiled model input nodes with
   `align_features_to_input_nodes()`.
4. Customize, train, and interpret the model with ordinary PyTorch,
   `customize_model()`, and `interpret_model()`.

## Installation

Install `edge2torch` from PyPI with:

```bash
pip install edge2torch
```

To run the `interpret_model()` function:

```bash
pip install "edge2torch[interpret]"
```

For optional `AnnData` support:

```bash
pip install "edge2torch[anndata]"
```

To install both optional extras:

```bash
pip install "edge2torch[all]"
```

## Minimal example

```python
import pandas as pd
import torch
from torch import nn

import edge2torch as e2t

edgelist = pd.DataFrame(
    {
        "source": ["entity_1", "entity_2", "hidden_1"],
        "target": ["hidden_1", "hidden_1", "output_1"],
    }
)

model, artifact = e2t.compile_graph(
    edgelist=edgelist,
    backend="feedforward",
)

data = pd.DataFrame(
    {
        "entity_1": [0.1, 0.2, 0.3],
        "entity_2": [1.0, 1.1, 1.2],
    }
)

x = e2t.align_features_to_input_nodes(
    data=data,
    artifact=artifact,
)

outputs = model(x)
```

The returned `model` is a standard PyTorch `nn.Module`. It can be trained,
wrapped, optimized, saved, and inspected with ordinary PyTorch tooling.

Optional convenience wrappers can be used for common downstream additions:

```python
customized_model = e2t.customize_model(
    model=model,
    activation=nn.ReLU(),
    head=nn.Linear(1, 1),
)

optimizer = torch.optim.Adam(customized_model.parameters(), lr=1e-3)
```

After training, the model can optionally be interpreted with Captum-based
attribution methods:

```python
feature_attributions = e2t.interpret_model(
    model=customized_model,
    artifact=artifact,
    data=data,
    target="features",
    method="IntegratedGradients",
)
```

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

## Main public API

The current public API is centered on:

- `compile_graph()`
- `align_features_to_input_nodes()`
- `customize_model()`
- `interpret_model()`

`CompileArtifact` is also exported as a public type because it is returned by
`compile_graph()` and consumed by helper functions. Its stable user-facing
fields are `backend` and `feature_names`; other fields expose compilation
internals for inspection and debugging and may change across releases.

`__version__` is exported for version reporting.

## Supported backends

`compile_graph()` currently supports:

- `feedforward`
- `recurrent`
- `graphnn`

Feature attribution is available through Captum-based methods. Feedforward
models also support broad node-level attribution. Recurrent and graph neural
network backends can be compiled and trained, while node-level interpretation
for these backends is planned for a future release.

## Documentation

The documentation website is the main source of package documentation.

It includes:

- installation instructions
- a getting-started tutorial
- backend documentation
- interpretation guidance
- API reference

Documentation: https://Thomas-Rauter.github.io/edge2torch/

## Citation

If you use `edge2torch` in research, please cite the software. Citation
metadata is available in [`CITATION.cff`](CITATION.cff).

## License

See `LICENSE`.
