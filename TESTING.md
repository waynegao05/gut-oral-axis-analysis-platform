# Testing Guide

## Full Suite

The default tests require the current local research artifacts under `outputs/current_mainline_v2/`.

```powershell
python -m pytest -q
```

## Focused Checks

```powershell
python -m pytest -q tests/test_pipeline.py tests/test_app_validation.py
python -m pytest -q tests/test_temporal_topology_bridge.py
python -m pytest -q tests/test_research_data_validation.py
```

The temporal bridge tests verify:

- exact replay against saved split-specific consensus risks;
- current release and backend identifiers;
- train-only topology inference;
- fixed calibration-anchor context;
- complete inferred function-score and edge-weight outputs.

## Static Checks

```powershell
python -m compileall -q archive config experiments research src tests
git diff --check
```

Before publishing, also confirm that no `outputs/`, local environment, editor state, or model checkpoint is staged.
