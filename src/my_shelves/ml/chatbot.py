"""
chatbot.py

This module implements a semantic book search system powered by
vector embeddings and Google Cloud Vertex AI Matching Engine.

Pipeline Overview
-----------------
The system follows a 5-step pipeline to enable semantic search:

1. Embedding Storage (GCS)
   -----------------------
   Book embeddings are generated locally and stored in Google Cloud Storage (GCS).
   These embeddings represent the semantic meaning of each book.

   Example:
       local_file = create_vertex_vector_file(df)
       gcs_uri = upload_to_gcs(
           local_file=local_file,
           bucket_name="my-shelves-vector-bucket",
           destination_blob_name="books_index/books_embeddings.json",
       )

2. Vector Index Creation
   ----------------------
   A vector index is created in Vertex AI Matching Engine using the stored embeddings.
   This index enables efficient nearest-neighbor search.

   Example:
       index = create_vector_index()

3. Index Endpoint Creation
   ------------------------
   An endpoint is created to serve the index.
   This acts as a scalable and accessible search service.

   Example:
       endpoint = create_index_endpoint()

4. Index Deployment
   -----------------
   The index is deployed to the endpoint, making it available for querying.

   Example:
       deploy_index(index, endpoint)

5. Semantic Search Usage
   ----------------------
   A query is converted into an embedding and sent to the deployed endpoint
   to retrieve the most relevant books.

   Example:
       model = SentenceTransformer(MODEL_NAME)
       results = search_books(
           model,
           query="A fantasy book with magic, dragons and adventure",
           endpoint=endpoint,
           top_k=5,
       )

Conceptual Mapping
-----------------
- GCS (📚): Stores book embeddings
- Index (📖): Organizes embeddings for fast similarity search
- Endpoint (📞): Exposes the search service
- Deployment: Connects the index to the endpoint
- Query: Retrieves similar books using semantic understanding

Notes
-----
- Index creation and deployment are time-consuming operations and should
  typically be executed offline or during infrastructure setup.
- The endpoint remains active after deployment and can be reused for multiple queries.
- Costs may apply for storage, deployment, and query usage in GCP.

"""

import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import storage, aiplatform


MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = Path("vector_data")
OUTPUT_FILE = OUTPUT_DIR / "books_embeddings.json"
LOCATION = "europe-west1"
GCS_INDEX_DIR = "gs://my-shelves-vector-bucket/books_index/"

def create_vertex_vector_file(df: pd.DataFrame) -> Path:
    """
    Create a JSONL file compatible with Vertex AI Vector Search.

    Each line contains one vector datapoint.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    documents = df["document"].fillna("").astype(str).tolist()

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row, embedding in zip(df.itertuples(index=False), embeddings):
            record = {
                "id": str(row.book_id),
                "embedding": embedding.tolist(),
                "embedding_metadata": {
                    "book_id": str(row.book_id),
                    "title": str(row.title),
                    "document": str(row.document)[:1000],
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return OUTPUT_FILE

def upload_to_gcs(
    local_file: Path,
    bucket_name: str,
    destination_blob_name: str,
) -> str:
    """
    Upload the JSONL embeddings file to Google Cloud Storage.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file)

    return f"gs://{bucket_name}/{destination_blob_name}"

def create_vector_index(project_id) -> aiplatform.MatchingEngineIndex:
    """
    Create a Vertex AI Vector Search index from embeddings stored in GCS.
    """

    aiplatform.init(
        project=project_id,
        location=LOCATION,
    )

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="books-vector-index",
        contents_delta_uri=GCS_INDEX_DIR,
        dimensions=384,
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        index_update_method="BATCH_UPDATE",
    )

    return index

def create_index_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Create a public endpoint for querying the vector index.
    """

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="books-vector-endpoint",
        public_endpoint_enabled=True,
    )

    return endpoint


def deploy_index(
    index: aiplatform.MatchingEngineIndex,
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
) -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Deploy the index to the endpoint.
    """

    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id="books_deployed_index",
    )

    return endpoint

def search_books(
    query,
    endpoint,
    top_k=5,
    model="all-MiniLM-L6-v2"
    ):
    """
    Retrieve the most relevant books based on a natural language query
    using a vector similarity search.

    This function converts the input query into an embedding using the
    provided model, sends it to a deployed Vertex AI Matching Engine
    endpoint, and returns the top-k most similar books.

    Parameters
    ----------
    query : str
        Natural language query describing the type of book to search for
        (e.g., "a fantasy book with magic and dragons").

    endpoint : aiplatform.MatchingEngineIndexEndpoint
        Deployed Vertex AI Matching Engine endpoint used to perform
        nearest neighbor search.

    top_k : int, optional (default=5)
        Number of most similar results to retrieve.

    model : object
        Embedding model used to encode the query into a vector.
        Must expose a method such as `encode()` or equivalent.

    Returns
    -------
    list[dict]
        A list of search results, typically containing:
        - book_id (or datapoint_id)
        - similarity score / distance
        - optional metadata if stored in the index

    Example
    -------
    >>> model = SentenceTransformer(MODEL_NAME)
    >>> results = search_books(
    ...     model=embedding_model,
    ...     query="a dark mystery set in London",
    ...     endpoint=endpoint,
    ...     top_k=5
    ... )
    >>> print(results)
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = endpoint.find_neighbors(
        deployed_index_id="books_deployed_index",
        queries=[query_embedding],
        num_neighbors=top_k,
    )

    return results
