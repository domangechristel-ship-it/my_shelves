
from prefect import task, flow
from prefect.artifacts import create_table_artifact


PREFECT_FLOW_NAME="locations"


@task
def get_locations():
    return


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    get_locations_task = get_locations.submit()
    get_locations_task.result()


if __name__ == "__main__":
    train_flow()
