APP_NAME = "Gut-Oral Axis Prototype"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True

DEFAULT_MICROBE_WEIGHTS = {
    "Fusobacterium": 1.2,
    "Porphyromonas": 1.1,
    "Prevotella": 0.7,
    "Streptococcus": 0.6,
    "Lactobacillus": -0.4,
}

DEFAULT_CLINICAL_WEIGHTS = {
    "age": 0.02,
    "bmi": 0.03,
    "smoking": 0.4,
    "family_history": 0.5,
}

DEFAULT_METABOLITE_WEIGHTS = {
    "bile_acids": 0.15,
    "scfa": -0.12,
    "tryptophan_metabolism": 0.11,
}

RISK_THRESHOLDS = {
    "low": 0.6,
    "medium": 1.2,
}
