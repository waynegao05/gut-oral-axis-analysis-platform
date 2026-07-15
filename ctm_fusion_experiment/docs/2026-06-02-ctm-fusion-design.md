# CTM Fusion Parallel Experiment Design

## Goal

Build an isolated exploratory branch that compares concat fusion against a
small-sample survival-analysis adaptation of the Continuous Thought Machine
(CTM). The comparison must change only the multimodal fusion method.

The experiment uses the repository's `topology_v6` synthetic/noisy augmented
research dataset. Results are method-development evidence, not clinical claims.

## Isolation Boundary

All new code lives under `ctm_fusion_experiment/`. Existing `research/` code and
the locked web-facing Cox ensemble remain unchanged. The experiment reads the
existing CSV tables and reuses the survival-task semantics from `research.task`.

## Fair Comparison Protocol

For each outer cross-validation fold:

1. Split samples into outer train and test sets with event stratification.
2. Split the outer training samples into inner train and validation sets.
3. Train a graph-only GNN Cox model using inner-train samples and select its
   checkpoint using inner-validation c-index.
4. Freeze the selected graph encoder and export graph embeddings for inner
   train, inner validation, and outer test samples.
5. Fit preprocessing statistics on inner-train vectors only.
6. Train concat-Cox and CTM-Cox fusion models from the same frozen graph
   embeddings, clinical features, metabolomics features, labels, and split.
7. Evaluate both fusion models on the same outer-test samples.

The paired fold-level c-index difference is the primary comparison. The new
baseline is a fair experimental control; it is not expected to reproduce the
locked five-seed ensemble score directly.

## Components

### Graph Encoder

`models/graph_encoder.py` contains a compact GAT-based graph-only Cox encoder.
It accepts the same graph tensors used by the current mainline and exports a
fixed-dimensional pooled graph embedding. It intentionally excludes clinical
and metabolomics data.

### CTM

`models/ctm.py` implements the CTM mechanisms required for this experiment:

- shallow synapse MLP over the previous activated state and attention output
- neuron-level MLPs with private weights evaluated in parallel with `einsum`
- random neuron pairings for action and output synchronization
- recurrent exponentially decayed synchronization updates
- cross-attention from synchronization state to the three modality tokens
- one synchronization-derived representation per internal tick

This is a survival-analysis adaptation for a small dataset, not a byte-for-byte
copy of the paper's task-specific image models.

### Fusion Models

`models/baseline_concat.py` implements concat fusion followed by a Cox head.
`models/ctm_fusion.py` projects graph, clinical, and metabolomics vectors into
three tokens, runs CTM ticks, and emits one Cox risk score per tick.

### Losses And Evaluation

The baseline uses Cox partial-likelihood loss. CTM computes Cox loss at every
tick and optimizes the mean of:

- the tick with minimum Cox loss
- the most stable tick, measured by the smallest adjacent-tick risk change

For evaluation, the stable tick is selected per sample without labels. Reports
include fold-level c-index, mean and standard deviation, a paired t-test,
parameter counts, timing, selected-tick histograms, and modality attention.

## Verification

Unit tests cover Cox loss, c-index, NLM shape and neuron-specific behavior,
recursive synchronization, CTM output shapes, deterministic split identity,
and paired statistical summaries. A CPU smoke configuration runs one fold with
small limits and short epochs. The full five-fold run is prepared but not
executed in the CPU-only local environment.
