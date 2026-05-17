import os
from pathlib import Path

APP_NAME = "Gut-Oral Axis Prototype"
HOST = os.getenv("GOA_HOST", "127.0.0.1")
PORT = int(os.getenv("GOA_PORT", "8765"))
DEBUG = os.getenv("GOA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
USE_RELOADER = os.getenv("GOA_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}

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
