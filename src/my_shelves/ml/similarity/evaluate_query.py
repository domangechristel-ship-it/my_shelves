"""
Evaluation script for the similarity query interface.
Tests the multi-tiered retrieval strategy (Local Cache -> BigQuery -> Model).
"""

import logging

from my_shelves.ml.similarity.similarity_query import get_similar_books
from my_shelves.ml.similarity.evaluate_models import get_sample_book_ids


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """
    Main entry point for evaluating the unified similarity query interface.
    """
    logging.info("Starting Similarity Query Interface Evaluation...\n")

    # Subset for evaluation
    eval_n_rows = "10k"
    model_names = ["knn_sk", "knn_tf", "sota_tf", "sota_torch", "sota_mpnet"]
    num_samples = 3

    # Get sample book IDs using the utility from evaluate.py
    sample_ids = get_sample_book_ids(eval_n_rows, num_samples=num_samples)
    if not sample_ids:
        logging.error("Failed to retrieve sample book IDs for evaluation.")
        return

    logging.info(
        f"Evaluating query strategy on {eval_n_rows} for books: {sample_ids}\n"
    )

    results = {}

    for model in model_names:
        logging.info(f"--- Querying Model Strategy: {model} ---")
        model_results = {}
        for book_id in sample_ids:
            # This calls the tiered strategy in similarity_query.py:
            # CSV Cache -> BigQuery -> Model Inference
            similar = get_similar_books(book_id, model, eval_n_rows)
            model_results[book_id] = similar
        results[model] = model_results

    logging.info("\n--- Query Comparison Results ---")
    for book_id in sample_ids:
        logging.info(f"\nResults for book_id {book_id}:")
        for model in model_names:
            sim_list = results[model].get(book_id, [])
            logging.info(f"  {model.ljust(12)}: {sim_list}")

    logging.info("\nEvaluation of query interface complete.")


if __name__ == "__main__":
    main()
