
DATASET_ROOT = "data"
MODELS_ROOT = "models"


N_ROWS_MAP = {"10k": 10_000,
              "20k": 20_000,
              "50k": 50_000,
              "100k": 100_000,
              "150k": 150_000,
              "200k": 200_000,
              "all": None
}

N_ROWS_NAMES = N_ROWS_MAP.keys()

BASE_FEATURES = []
LOCATION_FEATURES = ["book_id", "country", "region"]
EMOTION_FEATURES = ['emotions', 'top_emotion',
                    'cont_labels', 'cont_best_label',
                    'roma_labels', 'roma_best_label',
                    'char_labels', 'char_best_label',
                    'main_labels', 'main_best_label',
                    'pace_labels', 'pace_best_label',
                    'sentiment']
