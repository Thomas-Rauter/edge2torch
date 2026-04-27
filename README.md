# kpnn

Compile prior-knowledge biological graphs into minimally opinionated PyTorch
models, then map trained models back to interpretable node-level biology.

## Overview

`kpnn` is a graph-to-model compiler plus model-to-interpretation bridge for
knowledge-primed neural networks in biology.

The package is built around two ideas:

1. A prior-knowledge biological network can be compiled into a PyTorch model.
2. A trained PyTorch model can be mapped back to biological entities for
   interpretation.

The main goal is to stay as unopinionated as possible about the actual neural
network design. `kpnn` handles graph compilation, skip-aware structural
execution, and interpretation mapping. Neural network details such as
activations, output heads, losses, optimizers, and training loops are left to
the user.

## Installation

`kpnn` is not yet available on PyPI.

At the moment, it can be installed directly from the GitHub repository. Since
the repository is currently private, installation requires access.

### Install from GitHub

```bash
pip install git+ssh://git@github.com/Thomas-Rauter/kpnn.git
```

### Development install

For development, clone the repository and install it in editable mode:

```bash
git clone git@github.com:Thomas-Rauter/kpnn.git
cd kpnn
pip install -e .
```

### Optional extras

Install development dependencies with:

```bash
pip install -e .[dev]
```

Install documentation dependencies with:

```bash
pip install -e .[docs]
```

Install optional `AnnData` support with:

```bash
pip install -e .[bio]
```


## License

See `LICENSE`.
