# Research Task Definition

## Current primary task

The current main research task in this repository is:

- right-censored survival risk prediction

This is not a plain binary classification task and not a generic regression task.

## Label semantics

The canonical label table must contain:

- `sample_id`
- `time`
- `event`

Semantics:

- `time`: observed follow-up duration
- `event = 1`: observed event happened at `time`
- `event = 0`: right-censored sample, event was not observed within follow-up

Any workflow that ignores censoring and directly treats the problem as standard binary classification or ordinary regression is considered scientifically incompatible with the current main task.

## Primary evaluation

The primary evaluation metric is:

- concordance index (`c-index`)

This metric is shared by:

- deep GNN survival models
- linear Cox baselines
- discrete-time logistic hazard baselines
- tabular MLP survival baselines

## What is not the main task

The following are not considered the primary paper-facing task unless explicitly introduced as separate branches:

- plain binary risk classification
- ordinal risk stratification without survival semantics
- pseudo-risk regression without explicit censoring handling

## Dataset note

The current mainline configuration points to `topology_v6` research tables. Based on file naming and repository history, this dataset should be treated as synthetic / expanded research data with noise rather than a final real-clinical benchmark.

This implies:

- metric values are useful for method development
- metric values are not sufficient for strong clinical claims
- all reporting should explicitly state the dataset version and synthetic nature
