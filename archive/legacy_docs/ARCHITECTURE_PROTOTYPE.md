# Architecture Overview

This repository follows the patent-aligned analysis chain:

1. Oral microbiome feature input
2. Data preprocessing and normalization
3. Microbial interaction graph construction
4. GNN-style graph representation
5. COX-style risk scoring
6. Rule-based pharmacological suggestion generation
7. Structured report output

Current implementation status:

- Prototype ready
- Web demo ready
- Modular pipeline ready
- Rule base is illustrative
- Model weights are deterministic defaults rather than trained parameters

Core files:

- `src/preprocess.py`
- `src/graph_builder.py`
- `src/gnn_encoder.py`
- `src/risk_model.py`
- `src/recommendation.py`
- `src/report.py`
- `src/pipeline.py`
- `modular_app.py`
