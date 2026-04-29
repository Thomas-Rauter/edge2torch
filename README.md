# kpnn

Compile prior-knowledge graphs into minimally opinionated PyTorch models
and map trained models back to interpretable named entities.

**Documentation:** https://Thomas-Rauter.github.io/kpnn/

## Overview

`kpnn` is a graph-to-model compiler plus model-to-interpretation bridge for
knowledge-primed neural networks (KPNNs).

The package is not tied to a single scientific domain. Its core abstraction is
general: a graph with named entities is compiled into a PyTorch model, trained
with ordinary PyTorch tools, and then interpreted back in the space of the
original graph-defined entities.

Biology is currently a main application area, and many examples use biological
entities such as genes, transcription factors, and kinases. But the same ideas
can also be used in other domains wherever prior knowledge can be expressed as
a graph, including chemistry.

The package is built around three main steps:

1. Compile a prior-knowledge graph into a backend-specific PyTorch model with
   `compile_graph()`.
2. Customize and train the compiled model with ordinary PyTorch or with
   `customize_model()`.
3. Interpret the trained model with `interpret_model()`.

## Installation

Install `kpnn` from PyPI with:

```bash
pip install kpnn
```

For optional `AnnData` support:

```bash
pip install "kpnn[bio]"
```

## Minimal example

```python
import pandas as pd

from kpnn.compile_graph import compile_graph
from kpnn.customize_model import customize_model
from kpnn.interpret_model import interpret_model

edgelist = pd.DataFrame(
    {
        "source": ["entity_1", "entity_2", "hidden_1"],
        "target": ["hidden_1", "hidden_1", "output_1"],
    }
)

model, artifact = compile_graph(
    edgelist=edgelist,
    backend="feedforward",
)

customized_model = customize_model(model=model)

result = interpret_model(
    model=customized_model,
    artifact=artifact,
    data=...,
    target="features",
    method="integrated_gradients",
)
```

## Package philosophy

`kpnn` is intentionally minimally opinionated.

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

- `kpnn` handles graph compilation
- PyTorch handles model training
- `kpnn` maps trained models back to interpretable named entities

## Main public API

The current public API is centered on:

- `compile_graph()`
- `customize_model()`
- `interpret_model()`

## Documentation

The documentation website is the main source of package documentation.

It includes:

- installation instructions
- a getting-started tutorial
- backend documentation
- interpretation guidance
- API reference

Documentation: https://Thomas-Rauter.github.io/kpnn/

## License

See `LICENSE`.
