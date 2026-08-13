# Topology V7 analytic internal-edge v4

V3 showed that removing the generated edge prior and asking a free neural edge
generator to relearn it from survival loss reduced both C-index and
time-dependent AUC.

The V7 generator creates each edge from a bounded interaction between two
logit-standardized node abundances. V4 fits only the fold-local abundance
standardization and one intercept/slope pair per canonical edge. These
parameters are stored as model buffers. At inference, the forward pass computes
all ten symmetric edge weights from node abundance; CSV edge weights, cached
structure, outcomes, group IDs, and provenance are not read.

Before defining a new development/audit protocol, the inference-swap diagnostic
loads the already trained V3 legacy baseline and replaces only its inference
edge source. This diagnostic does not retrain the survival model and does not
use outcomes to fit the analytic edge layer.

If the inference swap is non-destructive, the fresh V4 development protocol
compares only three candidates: the legacy edge baseline, analytic internal
edges, and analytic internal edges with a zero-initialized bounded linear site
residual. A noninferiority result is reported separately but cannot trigger an
audit; only the locked performance-improvement gate can generate the one-time
audit cohort.
