# Installation

## Install from PyPI

Install `kpnn` from PyPI with:

```bash
pip install kpnn
```

For optional `AnnData` support:

```bash
pip install "kpnn[bio]"
```

## Development installation

To work on the package locally, clone the repository and install it in
editable mode from the project root:

```bash
git clone git@github.com:Thomas-Rauter/kpnn.git
cd kpnn
pip install -e .
```

For optional `AnnData` support during development:

```bash
pip install -e .[bio]
```

## Optional dependency groups

Install development dependencies with:

```bash
pip install -e .[dev]
```

Install documentation dependencies with:

```bash
pip install -e .[docs]
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

A minimal smoke test is:

```bash
python -c "import kpnn; print('kpnn imported successfully')"
```
