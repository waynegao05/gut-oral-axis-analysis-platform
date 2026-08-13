# AC-ICAM outcome transfer V10

This experiment tests a new-information route rather than another residual
stack on the same V7 labels.

The public source is the Sidra-LUMC AC-ICAM colon cancer cohort. Published
tumor and adjacent-normal 16S tables are joined to public patient-level PFS
time and censoring status. Only the five genera already measured by V7 are
used, so the resulting source model can be applied without inventing
unobserved taxa.

The experiment is isolated from the main model:

1. Convert the published RData tables to patient-level genus abundances.
2. Exclude stage IV and incomplete PFS records, matching the published
   prognostic analysis scope.
3. Audit a small, predeclared ridge-Cox grid with five repeated CV seeds.
4. Separately test whether the published 41-genus MBR score can be distilled
   from only the five genera that V7 actually observes.
5. Permit V7 development screening only if the source signal passes the
   locked gate.
6. Keep the formal V7 cohort untouched unless a further +0.005 development
   gate passes.

Run the source audit:

```powershell
python -m experiments.topology_v7_ac_icam_transfer_v10.source_audit
python -m experiments.topology_v7_ac_icam_transfer_v10.mbr_distillation
python -m experiments.topology_v7_ac_icam_transfer_v10.development_transfer
```

Original sources:

- Nature Medicine article: https://www.nature.com/articles/s41591-023-02324-5
- Figshare 16S tables: https://figshare.com/articles/dataset/Supplementary_Data_AC-ICAM/16944775
- cBioPortal study: https://www.cbioportal.org/study/summary?id=coad_silu_2022
- cBioPortal data repository: https://github.com/cBioPortal/datahub/tree/master/public/coad_silu_2022

The source tissue is not the same as saliva or stool. A passing model remains
an auxiliary prior, not evidence that the cohorts are interchangeable.
