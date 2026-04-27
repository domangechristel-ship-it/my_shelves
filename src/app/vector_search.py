from functools import lru_cache

from google.cloud import aiplatform
from sentence_transformers import SentenceTransformer


PROJECT_ID = '151819310613' #"my-shelves-493916"
LOCATION = "europe-west1"
INDEX_ENDPOINT_ID = '6454049692161409024' #"books-vector-endpoint"
DEPLOYED_INDEX_ID = "books_deployed_index" #'6980794926703312896'

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache
def get_model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache
def get_endpoint():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )

    return aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=endpoint_name
    )


def search_similar_books(query: str, top_k: int = 5):
    model = get_model()
    endpoint = get_endpoint()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=top_k,
    )

    return results
