# V4-V6 development results

## Decision table

| Experiment | Development seed | Candidate | C-index | iAUC | iBrier | Decision |
|---|---:|---|---:|---:|---:|---|
| V4 | 20261012 | legacy precomputed edge GNN | 0.753503 | 0.818182 | 0.135609 | reference |
| V4 | 20261012 | linear analytic internal edge | 0.752543 | 0.817324 | 0.135960 | reject |
| V5 | 20261014 | legacy precomputed edge GNN | 0.742642 | 0.802625 | 0.138086 | reference |
| V5 | 20261014 | exact internal edge GNN | 0.743091 | 0.803126 | 0.137888 | replacement noninferiority passed; performance gate failed |
| V6 | 20261016 | exact internal edge Cox | 0.732292 | 0.794374 | 0.142928 | reference |
| V6 | 20261016 | exact internal edge Cox + horizon rank 0.2 | 0.731995 | 0.794003 | 0.143058 | reject |

The cohorts use different locked generation seeds, so absolute scores must not
be used for cross-row model ranking. Only within-seed paired differences are
valid.

## Internal edge conclusion

V5 reconstructs V7 sample-level edge weights inside `forward` from node
abundance. The minimum held-out-group edge R2 was 0.999833 and the maximum MAE
was 0.000055. External edge values, cached graph structure, time, event,
generation group, and sample identifiers do not affect the reconstructed edge
weights.

On the fresh V5 cohort, the exact internal edge candidate improved C-index in
all five outer groups:

`+0.000454 / +0.001205 / +0.000522 / +0.000013 / +0.000050`

The macro gains were +0.000449 C-index and +0.000501 iAUC, with iBrier improving
by 0.000198. This passed the locked replacement noninferiority gate but did not
pass the performance-improvement gate. The audit cohort was therefore not
generated, and no mainline promotion was made.

The learned buffers are a fold-local supervised emulator of the V7 edge
teacher. They remove the need for precomputed edges at inference, but they are
not evidence of independently discovered biological interactions.

## Ranking objective conclusion

The V6 prescreen used four inner validation groups and never evaluated outer
group 0. Horizon ranking weight 0.2 produced a small inner C-index gain of
0.000507 and was locked before the V6 cohort was generated.

On the fresh V6 cohort, full four-fold inner LOGO selected the refit epoch for
each outer group. The rank candidate then reduced macro C-index by 0.000297,
reduced iAUC by 0.000371, and increased iBrier by 0.000130. It improved only one
of five outer groups. The candidate was rejected and audit seed 20261017 was
not generated.

## Data ceiling

The deterministic latent-risk C-index varied materially across locked
development cohorts:

- V4 seed 20261012: 0.760835
- V5 seed 20261014: 0.758676
- V6 seed 20261016: 0.740779

This variation is larger than all tested model deltas. Repeated model tuning
cannot reliably recover a 0.75-0.76 score when the generated observable outcome
contains less rank information. The next performance claim requires either a
more stable, externally justified outcome-generation protocol or measured
right-censored follow-up.

## Public-data boundary

The local source catalog records original repository pages and prevents
patient-level joins or invented survival labels. Open paired oral-gut cohorts
from Russo and Uchida can support representation learning and diagnosis
validation, but they do not expose exact right-censored survival. Debelius
provides a fixed-horizon binary survival task rather than time and censoring.
ColoCare exposes disease-free survival only through an approved data-sharing
request.

Original records:

- https://www.ncbi.nlm.nih.gov/bioproject/PRJNA899104
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217490
- https://ddbj.nig.ac.jp/public/ddbj_database/dra/fastq/DRA012/DRA012322/
- https://zenodo.org/records/7690117
- https://www.ebi.ac.uk/ena/browser/view/PRJEB57580
- https://uofuhealth.utah.edu/huntsman/labs/colocare-consortium/data-sharing/sharing
