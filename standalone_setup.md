# Standalone Repository Setup Guide

To run each project in its own Git repository, follow these steps to manage the shared `trading-core` dependency.

## 1. Prepare `trading-core`
In the `trading-core` repository, the `setup.py` and `pyproject.toml` are already configured.
To install it in your environment:
```bash
cd trading-core
pip install -e .
```
This installs `trading-core` in "editable" mode across your system/venv.

## 2. Setting up Sub-Projects
In your separate repositories (`trading-pipeline`, `trading-engine`, `trading-api`), the `requirements.txt` now includes:
```text
-e ../trading_core
```
**Important**: If you move the folders, update the path after `-e` or simply install the core project first from its own folder.

## 3. Why This Works
By making `trading-core` a package:
- You no longer need `sys.path` hacks in your code.
- You can import `from trading_core.core.physics import renko` from **anywhere** on your machine.
- Each repository stays clean and only contains its specific logic.

## 4. Git Structure Suggestion
We recommend a structure where `trading-core` is a **Git Submodule** or a **Private Pip Package** if you move to a cloud environment.
