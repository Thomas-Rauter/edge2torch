#!/usr/bin/env bash
set -euo pipefail

# Publish the current distributions in dist/ to the real PyPI server.
#
# Usage (run from project root):
#   chmod +x dev/pypi_release.sh
#   export TWINE_USERNAME="__token__"
#   export TWINE_PASSWORD="pypi-..."
#   ./dev/pypi_release.sh
#
# Notes:
# - TWINE_PASSWORD must be a real PyPI API token, not a TestPyPI token.
# - This script does not build the package. It uploads the current files in
#   dist/.
# - PyPI does not allow re-uploading the same version.
# - Run dev/dry_run_testpypi_release.sh first before using this script.

echo "Step 1: Checking that dist/ exists"
if [ ! -d "dist" ]; then
    echo "Error: dist/ does not exist."
    echo "Build the package before releasing."
    exit 1
fi

echo "Step 2: Checking that dist/ contains distributions"
if ! ls dist/*.tar.gz dist/*.whl >/dev/null 2>&1; then
    echo "Error: dist/ does not contain both sdist and wheel files."
    echo "Expected files like:"
    echo "  dist/edge2torch-0.1.0.tar.gz"
    echo "  dist/edge2torch-0.1.0-py3-none-any.whl"
    exit 1
fi

echo "Step 3: Listing files that will be uploaded"
ls -lh dist/

echo "Step 4: Checking distribution metadata"
python -m twine check dist/*

echo "Step 5: Confirming release target"
echo "This will upload the files in dist/ to the REAL PyPI server."
echo "Repository: https://upload.pypi.org/legacy/"
read -r -p "Type 'release' to continue: " CONFIRM

if [ "$CONFIRM" != "release" ]; then
    echo "Release cancelled."
    exit 1
fi

echo "Step 6: Uploading to PyPI"
python -m twine upload dist/*

echo "Done. Release uploaded to PyPI."
