
from prefect import task, flow
from prefect.artifacts import create_table_artifact

from my_shelves.prepare.train import train_flow as prepare_flow
from my_shelves.ml.similarity.train import train_flow as similarity_flow
from my_shelves.ml.location.train import train_flow as locations_flow
from my_shelves.ml.classification.train import train_flow as emotions_flow


PREFECT_FLOW_NAME="train-all"


@flow(name=PREFECT_FLOW_NAME)
def train_flow(n_rows: str = "20k"):
    prepare_task = prepare_flow()

    locations_task = locations_flow(wait_for=[prepare_task])

    emotions_task = emotions_flow(n_rows=n_rows, wait_for=[locations_task])

    similarity_task = similarity_flow(n_rows=n_rows, wait_for=[emotions_task])


if __name__ == "__main__":
    n_rows = "50k"
    train_flow(n_rows=n_rows)
