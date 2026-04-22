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

from fastapi import FastAPI, HTTPException
from my_shelves.utils.bigquery import get_book

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
