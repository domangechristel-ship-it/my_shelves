"""
Configuration parameters for similarity models and feature column definitions.
"""

DATASET_ROOT = "data"
MODELS_ROOT = "models/similarity"
N_ROWS_NAMES = ["10k", "20k", "50k", "100k", "150k", "200k", "all"]

NUM_COLS = [
    "n_votes",
    "read_duration",
    "average_rating",
    "num_pages",
    "ratings_count",
    "total_shelves_count"
]

_BASE_CAT_COLS = ["is_series", "author_names"]

CLASSIFICATION_COLS = [
    'emotions', 'content_intensity',
    'romance_heat_level', 'character_type',
    'main_themes', 'pace', 'sentiment'
]

LOCATION_COLS = ["country", "region"]

CAT_COLS = _BASE_CAT_COLS + CLASSIFICATION_COLS + LOCATION_COLS
