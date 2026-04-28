"""
API module for the My Shelves application.

Exposes a FastAPI app with the following endpoints:
    - GET /      : health check
    - GET /read  : retrieve a book description from BigQuery by book_id

Usage
-----
Run locally with Uvicorn:
    uvicorn my_shelves.api.api:app --reload

Or via Docker:
    docker run -e PORT=8000 -p 8080:8000 api:dev
"""

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from my_shelves.utils.bigquery import get_book, get_country_counts, get_books,get_id_by_country
from my_shelves.utils.bigquery import get_title, get_id_by_emotion, get_id_by_content_intensity
from my_shelves.utils.bigquery import get_id_by_romance_heat_level, get_id_by_character_type, get_id_by_main_themes
from my_shelves.utils.bigquery import get_id_by_pace, get_id_by_sentiment
from my_shelves.ml.similarity.main import get_similarity
from my_shelves.api.vector_search import search_similar_books

app = FastAPI()

@app.get('/')
def index():
    """Health check endpoint."""
    return {'ok': True}

@app.get('/read')
def read_book(book_id: int = 6668764):
    """
    Retrieve a book's information from BigQuery.

    Parameters
    ----------
    book_id : int, optional
        The unique identifier of the book (default: 6668764).

    Returns
    -------
    dict
        Dictionary containing the book information.

    Raises
    ------
    HTTPException
        If no book is found for the given book_id.
    """
    book = get_book(book_id)

    if book.empty:
        raise HTTPException(status_code=404, detail="Book not found")

    row = book.iloc[0]

    return {
        'book_id': str(row['book_id']),
        'description': str(row['description']),
        'publication_year': str(row['publication_year']),
        'image_url': str(row['image_url']),
        'url': str(row['url']),
        'average_rating': str(row['average_rating']),
        'title': str(row['title']),
        'num_pages': str(row['num_pages']),
        'series': str(row['series'])
    }

@app.get('/country')
def read_country_counts():
    """
    Retrieve aggregated book counts per country.

    This endpoint queries the underlying data source (e.g., BigQuery)
    to obtain the number of books associated with each country, along
    with related geographic metadata.

    Returns
    -------
    list[dict]
        A list of dictionaries where each dictionary represents a country
        and contains fields such as:
        - country : str
            The country name.
        - capital_latlng : list[float] or str
            Latitude and longitude of the country's capital.
        - count_books : int
            Number of books associated with the country.

    Notes
    -----
    The data is returned in JSON format using pandas `to_dict(orient="records")`,
    making it directly usable in frontend applications (e.g., Streamlit maps).
    """
    country_count = get_country_counts()

    return country_count.to_dict(orient="records")

@app.get("/books")
def read_books(
    book_id_list: list[int] = Query(...),
    nbr_rows: int | None = Query(10)
):
    """
    Retrieve books from BigQuery based on a list of book IDs.

    Parameters
    ----------
    book_id_list : list[int]
        list of book IDs to retrieve.
    nbr_rows : int | None, optional
        Maximum number of rows to return. If None, all matching rows are returned.

    Returns
    -------
    list[dict]
        List of books as dictionaries.
    """
    df = get_books(book_id_list=book_id_list, nbr_rows=nbr_rows)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No books found for the provided book_id_list."
        )

    # Remplacer les NaN par None pour avoir un JSON valide
    df = df.where(df.notna(), None)
    df = df.astype(object).where(pd.notnull(df), None)

    return df.to_dict(orient="records")

@app.get("/books/by-country")
def read_book_ids_by_country(country: str):
    """
    Retrieve the list of book IDs associated with a given country.

    Parameters
    ----------
    country : str
        Name of the country for which to fetch book IDs.

    Returns
    -------
    list[int]
        List of book IDs corresponding to the specified country.
        Returns an empty list if no books are found.

    """
    book_ids = get_id_by_country(country)
    return book_ids

@app.get("/books/similar")
def get_similar_book(book_id: str, model_name: str = None) -> list[int]:
    """
    Retrieve a list of similar books based on a given book_id.

    Parameters
    ----------
    book_id : int
        The unique identifier of the book to find similarities for.

    Returns
    -------
    list[dict]
        A list of IDs of similar books.
    """

    book_ids = get_similarity(int(book_id), model_name=model_name, n_rows="100k")
    return book_ids

@app.get("/books/chat-books")
def read_chat_books(query: str, top_k: int = 5):
    """
    Return book IDs from Vertex AI Vector Search.
    """
    try:
        results = search_similar_books(query=query, top_k=top_k)

        neighbors = results[0]

        return {
            "query": query,
            "top_k": top_k,
            "book_ids": [neighbor.id for neighbor in neighbors],
            "results": [
                {
                    "book_id": neighbor.id,
                    "distance": neighbor.distance,
                }
                for neighbor in neighbors
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while searching similar books: {str(e)}",
        ) from e
        
@app.get("/books/filter")
def filter_books(
    emotions: Optional[List[str]] = Query(default=None),
    main_themes: Optional[List[str]] = Query(default=None),
    content_intensity: Optional[str] = None,
    romance_heat_level: Optional[str] = None,
    character_type: Optional[str] = None,
    pace: Optional[str] = None,
    sentiment: Optional[str] = None,
):
    """
    Multi-criteria filtering with AND logic.
    Multi-select fields use OR internally, AND across categories.
    """

    book_ids = None

    def intersect(new_ids):
        nonlocal book_ids
        new_ids = set(new_ids)
        book_ids = new_ids if book_ids is None else book_ids & new_ids

    # 🔹 Multi-select → union puis intersection globale
    if emotions:
        ids = set()
        for emo in emotions:
            ids |= set(get_id_by_emotion(emo))  # OR
        intersect(ids)

    if main_themes:
        ids = set()
        for theme in main_themes:
            ids |= set(get_id_by_main_themes(theme))
        intersect(ids)

    # 🔹 Single filters → AND direct
    if content_intensity:
        intersect(get_id_by_content_intensity(content_intensity))

    if romance_heat_level:
        intersect(get_id_by_romance_heat_level(romance_heat_level))

    if character_type:
        intersect(get_id_by_character_type(character_type))

    if pace:
        intersect(get_id_by_pace(pace))

    if sentiment:
        intersect(get_id_by_sentiment(sentiment))

    return list(book_ids) if book_ids else []

# @app.get("/read/title")
# def search_books_by_title(title: str):

#     df = get_title(title)

#     if df.empty:
#         raise HTTPException(
#             status_code=404,
#             detail="No books found for the provided title."
#         )

#     # Remplacer les NaN par None pour avoir un JSON valide
#     df = df.where(df.notna(), None)
#     df = df.astype(object).where(pd.notnull(df), None)

#     return df.to_dict(orient="records")
