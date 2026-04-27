
import os
import pandas as pd
from prefect import task, flow
from prefect.artifacts import create_table_artifact
from my_shelves.ml.classification.params import OUTPUT_FOLDER, DATASET_ROOT
from my_shelves.ml.location.location import Location

PREFECT_FLOW_NAME="locations"

OUTPUT_FOLDER = "data/location"



@task
def get_locations(X: pd.DataFrame, n_rows: str = "10k") -> str:
    output_csv = f"{OUTPUT_FOLDER}/locations_{n_rows}.csv"

    feature_loc = Location()
    df_locations = feature_loc.get_locations(X)

    df_locations.to_csv(output_csv, index=False)

    return output_csv


@flow(name=PREFECT_FLOW_NAME)
def train_flow(n_rows: str = "10k"):
    df = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv",
                     # nrows=100, # only for local tests
                     usecols=["book_id", "description", "review_text"])
    X = df.copy()

    X.dropna(inplace=True)
    get_locations_task = get_locations.submit(X, n_rows=n_rows)
    get_locations_task.result()


if __name__ == "__main__":
    n_rows  = "20k"
    train_flow(n_rows=n_rows)
