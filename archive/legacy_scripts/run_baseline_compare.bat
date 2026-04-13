@echo off
setlocal
python -m research.baseline_compare --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
endlocal
