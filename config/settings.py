APP_NAME = "Gut-Oral Axis Prototype"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True

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
