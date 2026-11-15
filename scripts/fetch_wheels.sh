#!/usr/bin/env bash
# Pre-download the Linux wheels the Dockerfile installs offline (--no-index) into ./wheels.
# CPU-only torch comes from the PyTorch index; everything else from PIP_INDEX_URL (default PyPI).
# Use a mirror if PyPI is slow, e.g.:  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple scripts/fetch_wheels.sh
# Intel/AMD Docker host:  ARCH=x86_64 scripts/fetch_wheels.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${ARCH:-aarch64}"
INDEX="${PIP_INDEX_URL:-https://pypi.org/simple}"
PY="${PYTHON:-python3}"

plat=(--platform "linux_${ARCH}" --platform any)
for p in manylinux_2_28 manylinux2014 manylinux_2_17; do plat+=(--platform "${p}_${ARCH}"); done
common=(--dest wheels --only-binary=:all: "${plat[@]}" --python-version 3.12 --implementation cp --retries 10 --timeout 120)

mkdir -p wheels
"$PY" -m pip download "${common[@]}" --no-deps --index-url https://download.pytorch.org/whl/cpu torch
torch_version="$(basename wheels/torch-*.whl | cut -d- -f2)"
"$PY" -m pip download "${common[@]}" --index-url "$INDEX" --find-links wheels \
  "torch==${torch_version}" fastapi httpx uvicorn pydantic-settings gliner
