# Topology V7 exact internal-edge v5

V4 established that fold-local linear edge distillation was close to the
precomputed V7 edge input, but its held-out edge R2 of about 0.87 perturbed
training enough to miss the replacement gate.

V5 uses the actual bounded bilinear function family that generated V7 edges:

`clip(center + amplitude * tanh(b0 + b1*z_src + b2*z_dst + b3*z_src*z_dst))`

Each canonical edge has six frozen parameters. They are fitted only from the
current training groups' abundance and edge targets. Time, event, sample ID,
generation group, provenance, and hidden outcome variables are not inputs. At
inference, the model reads node abundance and computes all symmetric edge
weights inside `forward`; the external edge column and cached graph structure
are ignored.

The workflow first enforces an outcome-free held-out edge fidelity gate of
R2 >= 0.999. A fresh nested leave-one-group-out development cohort is generated
only under seed 20261014. The one-time audit cohort remains ungenerated unless
the internal-edge model passes the locked performance-improvement gate.
