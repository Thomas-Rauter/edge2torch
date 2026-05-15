#!/usr/bin/env bash
set -euo pipefail

# Release helper for TestPyPI.
#
# Usage (run from project root):
#   chmod +x ./dev/dry_run_testpypi_release.sh
#   export TWINE_USERNAME="__token__"
#   export TWINE_PASSWORD="pypi-..."
#   ./dev/dry_run_testpypi_release.sh
#
# Notes:
# - TWINE_PASSWORD must be a TestPyPI API token, not a real PyPI token.
# - TestPyPI does not allow re-uploading the same version.
# - For test releases, use versions such as 0.1.0rc1, 0.1.0rc2, etc.
# - This script creates and deletes a temporary virtual environment.

TEST_ENV="$(mktemp -d /tmp/edge2torch-test.XXXXXX)"
START_DIR="$(pwd)"

cleanup() {
    cd "$START_DIR"
    rm -rf "$TEST_ENV"
}

trap cleanup EXIT

echo "Step 1: Running local quality checks"
ruff check .
ruff format --check .
mypy src
pytest

echo "Step 2: Removing old build artifacts"
rm -rf dist build *.egg-info src/*.egg-info

echo "Step 3: Installing build tools"
python -m pip install --upgrade build twine

echo "Step 4: Building package"
python -m build

echo "Step 5: Checking distribution"
python -m twine check dist/*

echo "Step 6: Uploading to TestPyPI"
python -m twine upload --repository testpypi dist/*

echo "Step 7: Creating clean install test environment"
python -m venv "$TEST_ENV"

# shellcheck disable=SC1091
source "$TEST_ENV/bin/activate"

python -m pip install --upgrade pip

echo "Step 8: Installing edge2torch from TestPyPI"
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "edge2torch[interpret]"

echo "Step 9: Running installed-package smoke test outside repository"
cd /tmp

python - <<'PY'
import pandas as pd
import torch

import edge2torch
from edge2torch import (
    align_features_to_input_nodes,
    compile_graph,
    customize_model,
    interpret_model,
)

print(edge2torch.__version__ if hasattr(edge2torch, "__version__") else "ok")

edgelist = pd.DataFrame(
    {
        "source": ["feature_a", "feature_b", "hidden"],
        "target": ["hidden", "hidden", "prediction"],
    }
)

model, artifact = compile_graph(edgelist, quiet=True)

data = pd.DataFrame(
    {
        "feature_b": [2.0, 4.0],
        "feature_a": [1.0, 3.0],
    }
)

x = align_features_to_input_nodes(data, artifact)
customized_model = customize_model(model)

with torch.no_grad():
    y = customized_model(x)

feature_attributions = interpret_model(
    model=customized_model,
    artifact=artifact,
    data=x,
    target="features",
    method="IntegratedGradients",
    quiet=True,
)

print("feature_names:", artifact.feature_names)
print("model_output_shape:", tuple(y.shape))
print("feature_attributions_shape:", feature_attributions.shape)
print("TestPyPI install smoke test passed.")
PY

echo "Done. Temporary environment removed."
