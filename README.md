# Installation

## Environment

```bash
pip install uv
```

## Install dependencies

```bash
uv sync
```

## PyTorch special treatment

PyTorch is not yet fully supported for Python 3.13.

[PyTorch Issue #130249](https://github.com/pytorch/pytorch/issues/130249)

Thus, we need to install it manually.

```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```
