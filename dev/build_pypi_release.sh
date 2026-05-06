#!/usr/bin/env bash
set -euo pipefail

# Build final distributions for the real PyPI release.
#
# Usage (run from project root):
#   chmod +x dev/build_pypi_release.sh
#   ./dev/build_pypi_release.sh
#
# Notes:
# - Run dev/dry_run_testpypi_release.sh first with a release-candidate
#   version such as 0.1.0rc1.
# - Before running this script, bump the version to the final release version,
#   for example 0.1.0.
# - This script does not upload anything. It only creates and checks dist/.
# - Upload the resulting files with dev/pypi_release.sh.

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

echo "Step 5: Checking distribution metadata"
python -m twine check dist/*

echo "Step 6: Listing files prepared for release"
ls -lh dist/

echo "Done. Inspect dist/ and then run dev/pypi_release.sh."
