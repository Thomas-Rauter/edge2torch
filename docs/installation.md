# Installation

## Install from PyPI

Install the core `edge2torch` package from PyPI with:

```bash
pip install edge2torch
```

The core installation supports compiling sparse PyTorch models from edgelists,
aligning named input features, customizing compiled models, and training with
ordinary PyTorch.

## Optional interpretation support

`interpret_model()` uses Captum and is installed as an optional dependency.

To install `edge2torch` with interpretation support:

```bash
pip install "edge2torch[interpret]"
```

## Optional AnnData support

For optional `AnnData` input support:

```bash
pip install "edge2torch[anndata]"
```

To install both interpretation and `AnnData` support:

```bash
pip install "edge2torch[all]"
```

## Development installation

To work on the package locally, clone the repository and install it in
editable mode from the project root:

```bash
git clone git@github.com:Thomas-Rauter/edge2torch.git
cd edge2torch
pip install -e .
```

For optional interpretation support during development:

```bash
pip install -e ".[interpret]"
```

For optional `AnnData` support during development:

```bash
pip install -e ".[anndata]"
```

For both optional interpretation and `AnnData` support:

```bash
pip install -e ".[all]"
```

## Optional dependency groups

Install development dependencies with:

```bash
pip install -e ".[dev]"
```

Install documentation dependencies with:

```bash
pip install -e ".[docs]"
```

## Notebook and documentation note

Some documentation notebooks use optional visualization tools such as
Graphviz. If a notebook requires Graphviz rendering, you may also need the
system-level Graphviz installation in addition to the Python package.

For example, on Ubuntu or Debian:

```bash
sudo apt install graphviz
```

## Verify the installation

A minimal core-installation smoke test is:

```bash
python -c "import edge2torch; print('edge2torch imported successfully')"
```

To verify interpretation support, install `edge2torch[interpret]` and run:

```bash
python -c "from edge2torch import interpret_model; print(interpret_model)"
```
