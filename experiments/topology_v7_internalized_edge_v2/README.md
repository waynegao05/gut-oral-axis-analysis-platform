# Topology V7 internalized-edge v2

This experiment preserves the existing V7 GNN architecture and replaces only
the source of edge weights.

The legacy edge values may be used as fold-local training targets, but they are
never passed to the model at inference. The forward pass computes symmetric
sample-level edge weights from node abundance, node function score, clinical
features, and metabolite features. Outcome, generation group, sample ID, and
provenance fields are prohibited edge inputs.

Version 1 changed both the graph architecture and the edge source. Its best new
candidate reached macro C-index 0.749613 versus 0.749870 for the paired legacy
baseline, so it did not pass the locked gate. Version 2 removes that
architecture confounding and uses a new development-generation seed.

The audit cohort is generated only if a non-baseline candidate passes every
development gate.
