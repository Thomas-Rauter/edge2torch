# edge2torch

Compile prior-knowledge graphs into minimally opinionated PyTorch models
and map trained models back to interpretable named entities.

![Graphical abstract of edge2torch](figures/graphical_abstract.svg)

## Overview

`edge2torch` is a graph-to-model compiler plus model-to-interpretation bridge for
knowledge-primed neural networks.

The package is **not tied to a specific scientific domain**. Its core
abstraction is general: a graph with named entities is compiled into a PyTorch
model, trained with ordinary PyTorch tools, and then interpreted back in the
space of the original graph-defined entities.

Biology is currently a main application area, and many examples in the
documentation use biological entities such as genes, transcription factors, and
kinases. But the same ideas can also be used in other domains wherever prior
knowledge can be expressed as a graph. Chemistry is one example of such a
setting.

The package is built around three main steps:

1. **Compile** a graph into a backend-specific PyTorch model with
   `compile_graph()`
2. **Customize and train** the compiled model with ordinary PyTorch or with
   `customize_model()`
3. **Interpret** the trained model with `interpret_model()`

## Main public API

The current public API is centered on:

- `compile_graph()`
- `customize_model()`
- `interpret_model()`

A minimal workflow looks like:

```python
import pandas as pd

from edge2torch.compile_graph import compile_graph
from edge2torch.customize_model import customize_model
from edge2torch.interpret_model import interpret_model

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

customized_model = customize_model(
    model=model,
)

result = interpret_model(
    model=customized_model,
    artifact=artifact,
    data=...,
    target="features",
    method="integrated_gradients",
)
```

## Package philosophy

`edge2torch` is intentionally **minimally opinionated**.

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

## Current backends

`edge2torch` currently implements three backends:

- `feedforward`
- `recurrent`
- `graphnn`

These backends share the same graph input format but differ in how graph
structure is translated into neural-network computation.

See the **Backends** page for details.

## Documentation guide

If you are new to the package, the best place to start is:

- **Getting started** for a full end-to-end example
- **Installation** for package setup and optional extras
- **Backends** for backend semantics and current support
- **Interpretation** for attribution targets, methods, and backend support
- **API reference** for function-level documentation

## Current scope

The package currently focuses on:

- graph compilation into PyTorch models
- optional post-compilation model customization
- feature- and node-level interpretation
- feedforward, recurrent, and graphnn backend support

Interpretation support is currently most complete for the feedforward backend.

## License

See `LICENSE`.
