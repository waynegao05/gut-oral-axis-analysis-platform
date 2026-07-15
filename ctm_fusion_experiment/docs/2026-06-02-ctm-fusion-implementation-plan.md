# CTM Fusion Parallel Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, reproducible concat-versus-CTM survival-fusion experiment with fold-local frozen graph embeddings.

**Architecture:** Each outer CV fold trains a graph-only GNN Cox encoder on fold-local data, freezes and exports embeddings, then trains two fusion heads on identical vectors. CTM-specific code is independent from orchestration so tensor behavior is directly testable.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Geometric, pandas, numpy, scikit-learn, scipy, PyYAML, pytest.

---

## File Map

- `models/graph_encoder.py`: graph-only GNN Cox pretraining model.
- `models/cox_head.py`: shared scalar Cox head.
- `models/baseline_concat.py`: concat control model.
- `models/ctm.py`: CTM synapse, NLM, synchronization, and recurrent core.
- `models/ctm_fusion.py`: three-token CTM survival wrapper.
- `utils/data_loader.py`: fold splitting, PyG dataset construction, frozen-vector scaling.
- `utils/losses.py`: baseline and per-tick Cox losses.
- `utils/metrics.py`: c-index and paired fold summary.
- `utils/reporting.py`: JSON and CSV artifacts.
- `train.py`: fold-local orchestration and CLI.
- `evaluate.py`: rebuild comparison summary from fold artifacts.
- `tests/`: focused unit and smoke coverage.

### Task 1: Test Core Survival Utilities

- [ ] Add tests for Cox partial-likelihood loss, c-index, stable tick selection,
      and paired summary behavior.
- [ ] Run the tests and verify they fail before utility modules exist.
- [ ] Implement `utils/losses.py` and `utils/metrics.py`.
- [ ] Run the focused tests and verify they pass.

### Task 2: Test CTM Core

- [ ] Add tests for neuron-level models, recursive synchronization, and CTM
      output shapes.
- [ ] Run the tests and verify they fail before CTM modules exist.
- [ ] Implement `models/ctm.py`, `models/cox_head.py`,
      `models/baseline_concat.py`, and `models/ctm_fusion.py`.
- [ ] Run the focused tests and verify they pass.

### Task 3: Test Fold-Local Data Handling

- [ ] Add tests proving deterministic disjoint fold splits and train-only
      vector standardization.
- [ ] Run the tests and verify they fail before data helpers exist.
- [ ] Implement `utils/data_loader.py`.
- [ ] Run the focused tests and verify they pass.

### Task 4: Build Fold Training Orchestration

- [ ] Implement compact graph-only GNN Cox pretraining and embedding export.
- [ ] Implement fusion-model fitting, evaluation, artifact writing, and CLI
      configuration loading.
- [ ] Add a smoke configuration with sample and epoch limits.
- [ ] Run syntax checks and the unit test suite.

### Task 5: Execute CPU Smoke Test And Document Commands

- [ ] Run one fold with `configs/smoke.yaml`.
- [ ] Verify that baseline and CTM artifacts plus comparison summaries exist.
- [ ] Document smoke and formal five-fold commands in `README.md`.
- [ ] Re-run tests and inspect `git status` without changing the existing
      mainline or local output artifacts.
