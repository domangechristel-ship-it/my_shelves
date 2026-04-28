"""
Evaluation script for comparing different book similarity models.
"""

import logging
import pandas as pd

from my_shelves.ml.similarity.models.factory import ModelFactory
from my_shelves.ml.similarity.params import DATASET_ROOT


# Configure logging for the evaluation script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_sample_book_ids(n_rows: str, num_samples: int = 5) -> list:
    """Loads the dataset and returns a sample of book_ids."""
    data_path = f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv"
    try:
        df = pd.read_csv(data_path)
        df = df.drop_duplicates(subset='book_id')
        if len(df) < num_samples:
            logging.warning(
                f"Dataset for {n_rows} has fewer than {num_samples} "
                "unique book_ids. Returning all available."
            )
            return df['book_id'].tolist()
        return df['book_id'].sample(num_samples, random_state=42).tolist()
    except FileNotFoundError:
        logging.error(f"Dataset not found at {data_path}. Cannot get sample book IDs.")
        return []


def main():
    """
    Main entry point for evaluating similarity models.
    """
    logging.info("Starting similarity model evaluation process...\n")

    eval_n_rows = "10k"  # Choose a subset for evaluation
    model_names = ["knn_sk", "knn_tf", "sota_tf", "sota_torch", "sota_mpnet"]
    num_neighbors = 10  # Number of similar books to retrieve

    # Get sample book IDs for evaluation
    sample_book_ids = get_sample_book_ids(eval_n_rows, num_samples=5)
    if not sample_book_ids:
        logging.error("No sample book IDs available for evaluation. Exiting.")
        return

    logging.info(f"Evaluating models on dataset size: {eval_n_rows}")
    logging.info(f"Sample book IDs for evaluation: {sample_book_ids}\n")

    all_model_results = {}

    for model_name in model_names:
        logging.info(f"--- Processing Model: {model_name} ---")
        model = ModelFactory.create_model(model_name)
        model.train_or_load(n_rows=eval_n_rows)  # Train or load the model
        all_model_results[model_name] = {
            book_id: model.get_similar(book_id, n_neighbors=num_neighbors)
            for book_id in sample_book_ids
        }

        logging.info(f"Finished processing {model_name}.\n")

    logging.info("\n--- Comparison Results ---")
    for book_id in sample_book_ids:
        logging.info(f"\nSimilar books for book_id: {book_id}")
        for model_name in model_names:
            similar_list = all_model_results[model_name].get(book_id, [])
            logging.info(f"  {model_name.ljust(12)}: {similar_list}")

    logging.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
