"""
Unified query interface for retrieving book similarities.
Implements Step 18: Cache -> BigQuery -> Model inference.
"""

import os
import ast
import logging
import pandas as pd

from my_shelves.ml.similarity.params import DATASET_ROOT
from my_shelves.ml.similarity.bigquery import get_book_similarity as get_from_bq


def get_similar_books(book_id: int, model_name: str, n_rows: str) -> list[int]:
    """
    Retrieves similar books for a given ID using a prioritized search strategy.

    Strategy:
    1. Local CSV Cache (Fastest)
    2. BigQuery Table (Cloud Storage)
    3. Model Inference (Real-time calculation)
    """
    # 1. Try local CSV cache
    cache_path = os.path.join(
        DATASET_ROOT, "similarity",
        f"similarities_{model_name}_{n_rows}.csv"
    )
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path)
            match = cache_df[cache_df['book_id'] == book_id]
            if not match.empty:
                logging.info(f"Retrieved similarity for {book_id} from local cache.")
                return ast.literal_eval(match['similar_books'].iloc[0])
        except Exception as e:
            logging.warning(f"Failed to read local cache: {e}")

    # 2. Try BigQuery
    bq_results = get_from_bq(book_id, model_name, n_rows)
    if bq_results:
        logging.info(f"Retrieved similarity for {book_id} from BigQuery.")
        return bq_results

    # 3. Use the Model (Lazy Import of Factory to minimize startup time)
    logging.info(f"Computing similarity for {book_id} using {model_name} model inference.")
    from my_shelves.ml.similarity.models.factory import ModelFactory
    try:
        model = ModelFactory.create_model(model_name)
        model.train_or_load(n_rows=n_rows)
        return model.get_similar(book_id)
    except Exception as e:
        logging.error(f"Model inference failed for book_id {book_id}: {e}")

    return []


if __name__ == "__main__":
    # Example usage
    print(f"Similar books for book_id 1: {get_similar_books('sota_mpnet', '10k', 1)}")
