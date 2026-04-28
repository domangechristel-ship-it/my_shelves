"""
Prefect orchestration script for training and caching similarity models.
"""
from prefect import flow, task
from prefect.artifacts import create_table_artifact

from my_shelves.ml.similarity.models.factory import ModelFactory


@task(name="Train Model")
def train_model_task(model_name: str, n_rows: str):
    """
    Prefect task to train or load a specific model.
    """
    print(f"\n--- Training/Loading Model: {model_name} ---")
    model = ModelFactory.create_model(model_name)
    model.train_or_load(n_rows=n_rows)

    create_table_artifact(
        key=f"trained-model-{model_name.replace("_", "-")}-{n_rows}",
        table=[
            {"Model Name": model_name,
             "Dataset Size": n_rows,
             "Status": "Trained/Loaded"
             }
        ],
        description=f"Model {model_name} trained/loaded for {n_rows} dataset."
    )
    return model_name


@task(name="Save Cache")
def save_cache_task(model_name: str, n_rows: str):
    """
    Prefect task to save the similarity cache for a model.
    """
    print(f"--- Saving Cache: {model_name} ---")
    model = ModelFactory.create_model(model_name)
    # Ensure the model is loaded with its artifacts before caching
    model.load(n_rows=n_rows)
    model.save_cache(n_rows=n_rows, batch_size=500)
    print(f"Cache saved for {model_name}.")

    create_table_artifact(
        key=f"cache-saved-{model_name.replace("_", "-")}-{n_rows}",
        table=[
            {"Model Name": model_name,
             "Dataset Size": n_rows,
             "Status": "Cache Saved"
             }
        ],
        description=f"Model {model_name} trained/loaded for {n_rows} dataset."
    )
    return model_name


@flow(name="Similarity Models Training Flow")
def training_flow():
    """
    Prefect flow that trains and caches all similarity models sequentially
    for a fixed dataset size of 10k.
    """
    n_rows = "10k"
    model_names = ["knn_sk", "knn_tf", "sota_tf", "sota_torch", "sota_mpnet"]

    print(f"Starting Prefect training flow for dataset size: {n_rows}\n")

    for model_name in model_names:
        trained_model = train_model_task(model_name, n_rows)
        save_cache_task(trained_model, n_rows)

    print(f"\n--- Finished all models for {n_rows} dataset size ---")


if __name__ == "__main__":
    training_flow()
