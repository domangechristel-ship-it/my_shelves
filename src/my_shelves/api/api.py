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
from my_shelves.utils.bigquery import get_book, get_country_counts, get_books

app = FastAPI()

@app.get('/')
def index():
    """Health check endpoint."""
    return {'ok': True}

@app.get('/read')
def read_book(book_id: int = 22077083):
    """
    Retrieve a book's information from BigQuery.

    Parameters
    ----------
    book_id : int, optional
        The unique identifier of the book (default: 22077083).

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
