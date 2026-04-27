
import datetime
import tensorflow as tf

from prefect import task, flow
from prefect.artifacts import create_table_artifact

from my_shelves.ml.similarity.models.knn_tf import SimilarityKNNTF
from my_shelves.ml.similarity.models.knn_sk import SimilarityKNNSK
from my_shelves.ml.similarity.models.sota_torch import SimilaritySotaTorch
from my_shelves.ml.similarity.models.sota_tf import SimilaritySotaTF
# from my_shelves.ml.similarity.models.base_similarity import SimilarityModel
from my_shelves.ml.similarity.params import N_ROWS_NAMES


PREFECT_FLOW_NAME="similarity"


@task
def train_knn_tf(n_rows: str = "10k") -> str:
    start = datetime.datetime.now()

    knn_tf_model = SimilarityKNNTF()
    knn_tf_model.train_or_load(n_rows=n_rows)

    end = datetime.datetime.now()
    duration = (end - start).total_seconds() / 60

    table = [
       {'n_rows': n_rows,
        'duration': duration
        }
    ]

    create_table_artifact(
        key="train-knn-tf",
        table=table,
        description= "# Similarity Model using KNN with ScikitLearn."
    )
    return "knn_tf_model training finished."


@task
def train_knn_sk(n_rows: str = "10k") -> str:
    start = datetime.datetime.now()

    knn_sk_model = SimilarityKNNSK()
    knn_sk_model.train_or_load(n_rows=n_rows)

    end = datetime.datetime.now()
    duration = (end - start).total_seconds() / 60

    table = [
       {'n_rows': n_rows,
        'duration': duration
        }
    ]

    create_table_artifact(
        key="train-knn-sk",
        table=table,
        description= "# Similarity Model using KNN with TensorFlow."
    )

    return "knn_sk_model training finished."


@task
def train_sota_tf(n_rows: str = "10k") -> str:
    start = datetime.datetime.now()

    sota_tf_model = SimilaritySotaTF()
    sota_tf_model.train_or_load(n_rows=n_rows)

    end = datetime.datetime.now()
    duration = (end - start).total_seconds() / 60

    table = [
       {'n_rows': n_rows,
        'duration': duration
        }
    ]

    create_table_artifact(
        key="train-sota-tf",
        table=table,
        description= "# Similarity Model using SOTA with TensorFlow."
    )
    return "sota_tf_model training finished."


@task
def train_sota_torch(n_rows: str = "10k") -> str:
    start = datetime.datetime.now()

    sota_torch_model = SimilaritySotaTorch()
    sota_torch_model.train_or_load(n_rows=n_rows)

    end = datetime.datetime.now()
    duration = (end - start).total_seconds() / 60

    table = [
       {'n_rows': n_rows,
        'duration': duration
        }
    ]

    create_table_artifact(
        key="train-sota-torch",
        table=table,
        description= "# Similarity Model using SOTA with PyTorch."
    )
    return "sota_torch_model training finished."


@task
def notify(msg: list[str]):
    for m in msg:
        print(m)


@flow(name=PREFECT_FLOW_NAME)
def train_flow(n_rows: str = "10k"):
    """
    Build the Prefect workflow for the `similarity` models.
    """
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    knn_tf_task = train_knn_tf.submit(n_rows=n_rows)
    knn_sk_task = train_knn_sk.submit(n_rows=n_rows, wait_for=[knn_tf_task])
    sota_tf_task = train_sota_tf.submit(n_rows=n_rows, wait_for=[knn_sk_task])
    sota_torch_task = train_sota_torch.submit(n_rows=n_rows, wait_for=[sota_tf_task])

    knn_tf_result = knn_tf_task.result()
    knn_sk_result = knn_sk_task.result()
    sota_tf_result = sota_tf_task.result()
    sota_torch_result = sota_torch_task.result()

    notify_task = notify.submit(msg=[knn_tf_result,
                                     knn_sk_result,
                                     sota_tf_result,
                                     sota_torch_result])
    notify_task.result()


if __name__ == "__main__":
    n_rows = "10k"
    train_flow(n_rows=n_rows)
