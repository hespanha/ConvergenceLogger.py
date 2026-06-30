# AGENTS.md

This repository contains a python package to monitor and visualize the convergence of machine learning algorithms by tracking statistics over time.

Start with [README.md](README.md) for environment setup, usage examples, and development conventions.

## Environment

- Use Python 3.12 on macOS. The repo notes that Torch 2.2.2 does not work with Python 3.13 on macOS.
- Create a virtual environment in the repo root and install dependencies with `pip install -r requirements.txt`.
- Run module scripts and smoke tests from the repository virtual environment at `.venv` (for example: `source activate`, which resolves to `.venv/bin/activate`).
- For development tools, install `pip install -e ".[dev]"`.
- Some notebooks and plotting workflows expect `ipywidgets` and `PyQt6`; do not assume they are available in headless test environments.

## Run And Test

- Primary automated check: `pytest -q tests/test_convergence_logger.py`
- Broader test run: `pytest -q`
- Type-checking and lint config live in [pyproject.toml](pyproject.toml). `pyright` is configured with extra paths for both experiment directories, and `ruff` uses those same source roots.

## Repository Layout

- `convergence_logger/`: source code for the convergence logging library with pluggable data sources.
- `tests/`: Unit tests for the convergence logging library.

## Project Conventions

- Keep changes local and research-oriented. Prefer small edits to existing scripts and helpers over introducing new framework structure.
- Preserve the existing script-first style. Much of the code is meant to be run from the repository root.
- The main package `convergence_logger` uses standard Python package imports. Example scripts import from `convergence_logger`.
- Prefer modern Python type annotations in new or edited code, such as built-in generics (`list[int]`, `dict[str, Any]`) and `X | Y` unions.
- Include class-level type annotations for all class variables. These annotations serve as a summary of the class's state and improve readability, even when variables are assigned in `__init__`.
- Keep docstrings synchronized with function/class signatures and type hints whenever code is edited. Include type annotations in docstring `Args` sections (e.g., `x (torch.Tensor): Description`).
- When editing tensor code, annotate tensor-creating operations with brief shape comments when practical, preferably on the same line as the computation (e.g., `x = torch.randn(2, 3)  # shape: (2, 3)`).
- Prefer dependency-based shape comments over ambiguous ellipsis notation; for example, use phrases like "shape: same as z.shape" when a tensor preserves another tensor's shape.
- In training entry points, preserve reproducibility patterns: seed Torch from parameters and reset datasource RNG state before loops.
- Keep training scripts headless-safe: guard interactive plotting with backend checks and avoid forcing interactive displays in non-interactive environments.
- Prefer device-aware tensor creation patterns (for example, `torch.rand_like(...)` or explicit `device=...`) to avoid accidental CPU/GPU mismatch.
- Prefer abstract base classes when they clarify a stable interface or remove meaningful duplication, but avoid adding ABC layers that only add boilerplate.
- When touching datasource APIs, keep `tests/test_datasources.py` in sync. The tests expect stable method names, output shapes, plotting buffers, and diagnostic keys.

## Agent Guidance

- Before editing, find the nearest controlling script, helper, or test for the requested behavior and work from there.
- After changes in `convergence_logger` datasources or shared plotting logic, run `pytest -q tests/test_datasources.py` before widening scope.
- Link to existing docs instead of copying long setup instructions into new files.
