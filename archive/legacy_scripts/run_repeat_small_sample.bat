@echo off
setlocal
python -m research.repeat_runs_cli --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
endlocal
