import os
from pathlib import Path

APP_NAME = "Gut-Oral Axis Prototype"
HOST = os.getenv("GOA_HOST", "127.0.0.1")
PORT = int(os.getenv("GOA_PORT", "8765"))
DEBUG = os.getenv("GOA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
USE_RELOADER = os.getenv("GOA_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}
WEB_MODEL_BACKEND = os.getenv("GOA_MODEL_BACKEND", "temporal_topology").strip().lower()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_MODEL_CONFIG_PATH = PROJECT_ROOT / "research_config_v2.yaml"
RESEARCH_MODEL_RELEASE_NAME = "cox_fixed_split_ensemble_v1"
RESEARCH_MODEL_RELEASE_NOTE = (
    "Locked web release for the conservative GNN + Cox mainline using the "
    "5-seed fixed-split ensemble."
)
RESEARCH_MODEL_CHECKPOINT_GLOB = "outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt"
RESEARCH_MODEL_EXPECTED_ENSEMBLE_SIZE = 5
RESEARCH_MODEL_ALLOW_FALLBACK = os.getenv("GOA_ALLOW_SINGLE_CHECKPOINT_FALLBACK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RESEARCH_MODEL_FALLBACK_CHECKPOINT = PROJECT_ROOT / "outputs/current_mainline_v2/research/best_model.pt"
RESEARCH_MODEL_MAX_REFERENCE_BATCH = 64
RESEARCH_MODEL_REFERENCE_CACHE = PROJECT_ROOT / "outputs" / "current_mainline_v2" / "web_research_model_reference_cache.json"
RESEARCH_MODEL_RELEASE_METRICS = {
    "split_seed": 42,
    "single_model_mean_c_index": 0.7387742292571311,
    "single_model_std_c_index": 0.002903875316886964,
    "ensemble_c_index": 0.7409744768991116,
    "tabular_best_c_index": 0.7395456853805641,
    "graph_perturbation_gap_min": 0.023192755616166405,
    "graph_perturbation_gap_max": 0.025424957062076037,
}

# Latest formal web backend. The older Cox bridge remains available in
# archive/legacy_web_backends/cox_ensemble_v1.py for local comparison and rollback.
TEMPORAL_TOPOLOGY_RELEASE_NAME = "temporal_topology_aft_cross_split_consensus_v1"
TEMPORAL_TOPOLOGY_RELEASE_NOTE = (
    "Cross-split consensus of the structure-aware GNN mainline and the "
    "five-seed temporal-topology AFT expert. Web topology is inferred from "
    "the submitted abundances, clinical variables, and metabolites."
)
TEMPORAL_TOPOLOGY_ROOT = PROJECT_ROOT / "outputs" / "current_mainline_v2" / "temporal_independent_v3"
TEMPORAL_TOPOLOGY_FULL_RISK_ROOT = PROJECT_ROOT / "outputs" / "current_mainline_v2" / "full_risk_head_refiner_v2"
TEMPORAL_TOPOLOGY_SPLIT_SEEDS = (42, 43)
TEMPORAL_TOPOLOGY_MODEL_SEEDS = (7, 21, 42, 123, 2026)
TEMPORAL_TOPOLOGY_DEVICE = os.getenv("GOA_TEMPORAL_DEVICE", "cpu").strip().lower()
TEMPORAL_TOPOLOGY_RELEASE_METRICS = {
    "evaluation_protocol": "mean_of_two_split_specific_held_out_tests",
    "selection_uses_test_labels": False,
    "consensus_alpha": 0.63,
    "mean_reference_test_c_index": 0.740333369578418,
    "mean_selected_test_c_index": 0.7570563484482322,
    "mean_test_c_index_delta": 0.01672297886981422,
    "mean_test_calibrated_cox_loss_delta": -0.013212326324814505,
    "split42_selected_test_c_index": 0.760865719845024,
    "split43_selected_test_c_index": 0.7532469770514404,
    "metric_scope_note": (
        "These are the mean split-specific held-out results, not a direct "
        "validation score for the deployment-time average of both splits."
    ),
    "deployment_scope_note": (
        "Held-out metrics used research-table topology and deterministic eight-sample "
        "evaluation batches. Web inference uses inferred topology and a fixed calibration "
        "anchor, so the held-out C-index must not be presented as direct web validation."
    ),
    "dataset_note": "topology_v6 is synthetic/noisy augmented research data.",
}
TEMPORAL_TOPOLOGY_INFERENCE_METRICS = {
    "protocol": "strict_split_train_ridge_evaluated_on_validation_and_test",
    "split42": {
        "function_score_mae": 0.1277855247282125,
        "function_score_r2": 0.33959964297996426,
        "edge_weight_mae": 0.15440188178524603,
        "edge_weight_r2": 0.31242098908295723,
    },
    "split43": {
        "function_score_mae": 0.1282800074900478,
        "function_score_r2": 0.3372923994984766,
        "edge_weight_mae": 0.15270936192840592,
        "edge_weight_r2": 0.3257498418334567,
    },
    "scope_note": (
        "This measures reconstruction of research-table topology from web-available "
        "inputs; inferred values are not laboratory measurements."
    ),
}

DEFAULT_MICROBE_WEIGHTS = {
    "Fusobacterium": 0.9,
    "Porphyromonas": 0.8,
    "Prevotella": 0.4,
    "Streptococcus": 0.2,
    "Lactobacillus": -0.5,
}

DEFAULT_CLINICAL_WEIGHTS = {
    "age": 0.008,
    "bmi": 0.015,
    "smoking": 0.25,
    "family_history": 0.3,
}

DEFAULT_METABOLITE_WEIGHTS = {
    "bile_acids": 0.10,
    "scfa": -0.15,
    "tryptophan_metabolism": 0.08,
}

RISK_THRESHOLDS = {
    "low": 1.4,
    "medium": 2.4,
}
