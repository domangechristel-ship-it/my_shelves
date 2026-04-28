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
from my_shelves.utils.bigquery import get_book, get_country_counts, get_books,get_id_by_country
from my_shelves.ml.similarity.similarity_query import get_similar_books


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

    book_ids = get_similar_books(int(book_id), model_name=model_name, n_rows="20k")
    return book_ids
