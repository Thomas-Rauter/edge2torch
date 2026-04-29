# Interpretation

`kpnn` provides `interpret_model()` as the model-to-interpretation bridge of
the package.

It computes attribution scores for a compiled KPNN model and maps them back to
the named entities preserved by the compilation artifact.

## Overview

Interpretation in `kpnn` is built around three ideas:

1. a compiled model is still an ordinary PyTorch model
2. attribution is computed with Captum-based methods
3. returned results are mapped back to graph-defined names through the
   `KPNNArtifact`

This means that `kpnn` does not try to replace PyTorch or Captum. Instead, it
provides a thin, validated interface that connects:

- compiled KPNN models
- supported attribution methods
- user-facing feature and node names

## Main interpretation API

The main public entry point is:

```python
from kpnn.interpret_model import interpret_model
```

A typical call looks like:

```python
result = interpret_model(
    model=model,
    artifact=artifact,
    data=data,
    target="features",
    method="integrated_gradients",
)
```



## Interpretation targets

`interpret_model()` currently supports two targets:

- `target="features"`
- `target="nodes"`

### Feature target

`target="features"` means:

- compute attributions for input features
- return the results indexed by the original feature names from the artifact

The returned object is one pandas DataFrame with:

- rows = examples
- columns = feature names

### Node target

`target="nodes"` means:

- compute attributions for internal named graph entities
- return the results grouped by structural units such as layers

The returned object is a dictionary:

- keys = layer names such as `"layer_1"` or `"layer_2"`
- values = pandas DataFrames

Each returned DataFrame has:

- rows = examples
- columns = node names for that layer

## Supported methods

The currently supported methods are:

- `integrated_gradients`
- `layer_conductance`
- `layer_integrated_gradients`

## Supported target / method combinations

Not every method is compatible with every target.

Current supported combinations are:

- `target="features"`
  - `method="integrated_gradients"`

- `target="nodes"`
  - `method="layer_conductance"`
  - `method="layer_integrated_gradients"`

Unsupported combinations raise a `KPNNError`.

## Backend-dependent interpretation support

Interpretation support also depends on the compiled backend.

| Backend     | `target="features"` + `integrated_gradients` | `target="nodes"` + `layer_conductance` | `target="nodes"` + `layer_integrated_gradients` |
|-------------|----------------------------------------------|----------------------------------------|-------------------------------------------------|
| feedforward | yes                                          | yes                                    | yes                                             |
| recurrent   | yes                                          | no                                     | no                                              |
| graphnn     | yes                                          | no                                     | no                                              |

## Why feature interpretation is more general

Feature-level interpretation is currently supported for all implemented
backends because it only requires that the compiled model behaves as a normal
differentiable PyTorch model with a standard input-output interface.

Node-level interpretation is more backend-specific, because it depends on how
internal model structure is represented and how attribution results can be
mapped back to named internal entities.

## Input data formats

`interpret_model()` currently accepts:

- `pandas.DataFrame`
- `torch.Tensor`
- optional `anndata.AnnData`

The input data must match the compiled feature space defined by the artifact.

In particular:

- DataFrame columns must include the required feature names
- tensor inputs must have the correct number of feature columns
- AnnData inputs must have the correct number of variables

## Basic feature interpretation example

A minimal feature-level example is:

```python
import pandas as pd
from kpnn.compile_graph import compile_graph
from kpnn.interpret_model import interpret_model

edgelist = pd.DataFrame(
    {
        "source": ["gene_1", "gene_2"],
        "target": ["pathway_1", "pathway_1"],
    }
)

model, artifact = compile_graph(
    edgelist=edgelist,
    backend="feedforward",
)

data = pd.DataFrame(
    {
        "gene_1": [0.1, 0.2, 0.3],
        "gene_2": [1.0, 1.1, 1.2],
    },
    index=["cell_1", "cell_2", "cell_3"],
)

feature_attr = interpret_model(
    model=model,
    artifact=artifact,
    data=data,
    target="features",
    method="integrated_gradients",
)
```

The returned object is one DataFrame with per-example feature attributions.

## Basic node interpretation example

A minimal node-level example is:

```python
import pandas as pd
from kpnn.compile_graph import compile_graph
from kpnn.interpret_model import interpret_model

edgelist = pd.DataFrame(
    {
        "source": ["gene_1", "gene_2", "pathway_1"],
        "target": ["pathway_1", "pathway_1", "output_1"],
    }
)

model, artifact = compile_graph(
    edgelist=edgelist,
    backend="feedforward",
)

data = pd.DataFrame(
    {
        "gene_1": [0.1, 0.2],
        "gene_2": [1.0, 1.1],
    },
    index=["cell_1", "cell_2"],
)

node_attr = interpret_model(
    model=model,
    artifact=artifact,
    data=data,
    target="nodes",
    method="layer_conductance",
)
```

The returned object is a dictionary mapping layer names to per-layer
attribution DataFrames.

## Pseudo nodes and user-facing outputs

Pseudo nodes are internal compiler-generated nodes used in the feedforward
backend for skip-edge expansion.

They may participate in internal computation, but they are not part of the
user-facing interpretation space.

For that reason:

- pseudo nodes may exist in the compiled artifact
- pseudo nodes may participate internally in attribution computation
- pseudo nodes are filtered out of returned node-level interpretation outputs

This means the user-facing results contain only the original named graph
entities, not compiler-generated pseudo nodes.

## Validation behavior

`interpret_model()` validates:

- `target`
- `method`
- `target / method` compatibility
- backend support
- input container type
- input feature compatibility
- artifact structure

This validation is intentionally strict so that unsupported or ambiguous
interpretation requests fail early and clearly.

## Design principle

`interpret_model()` is intentionally narrow in scope.

It is responsible for:

- validating interpretation inputs
- standardizing accepted data containers into tensors
- dispatching to a supported Captum-based interpretation path
- mapping outputs back to artifact-defined names

It is not responsible for:

- training the model
- choosing the best interpretation method automatically
- plotting
- aggregating results across samples by default
- ranking entities by importance
- downstream biological or domain-specific analysis

Those remain outside the core interpretation API.

## Relationship to the rest of the package

`kpnn` is organized around three main public layers:

- `compile_graph()`  
  graph-to-model compiler

- `customize_model()`  
  optional post-compilation architectural customization

- `interpret_model()`  
  model-to-interpretation bridge

A typical workflow is therefore:

```python
model, artifact = compile_graph(...)
customized_model = customize_model(model=..., ...)
train customized_model with ordinary PyTorch
result = interpret_model(
    model=customized_model,
    artifact=artifact,
    data=data,
    target="features",
    method="integrated_gradients",
)
```

## Current limitations

At the moment:

- feature-level interpretation is available for all current backends
- node-level interpretation is implemented only for the feedforward backend

This may expand in the future as backend-specific internal interpretation
semantics are defined more fully.
