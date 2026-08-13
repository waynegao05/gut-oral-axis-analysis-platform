# Local Public Cohort Data

Store downloaded public or approved controlled-access cohort files here:

```text
data/public/<dataset-id>/raw/
data/public/<dataset-id>/processed/
data/public/<dataset-id>/cohort_manifest.json
```

This directory remains the unchanged source-data area. Its contents are
ignored by Git by default so that large files, licensed files, and
participant-level data are not accidentally published.

`topology_v7` learns a generated oral-gut distribution from the processed
Russo cohort, but never writes generated records back into this directory. The
source cohort must remain available for separate real-data evaluation.
