
import os
import pandas as pd
from prefect import task, flow
from prefect.artifacts import create_table_artifact
from my_shelves.ml.classification.params import OUTPUT_FOLDER
from my_shelves.ml.location.location import Location

PREFECT_FLOW_NAME="locations"


@task
def get_locations(X: pd.DataFrame, n_rows: str = "10k") -> str:
    output_csv = f"{OUTPUT_FOLDER}/locations_{n_rows}.csv"

    feature_loc = Location()
    df_locations = feature_loc.get_locations(X)

    df_locations.to_csv(output_csv, index=False)

    return output_csv


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    get_locations_task = get_locations.submit()
    get_locations_task.result()


if __name__ == "__main__":
    train_flow()
