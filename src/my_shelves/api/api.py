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

from fastapi import FastAPI
from my_shelves.utils.bigquery import get_book

app = FastAPI()

@app.get('/')
def index():
    """Health check endpoint."""
    return {'ok': True}

@app.get('/read')
def read_book(book_id: int = 22077083):
    """
    Retrieve a book's description from BigQuery.

    Parameters
    ----------
    book_id : int, optional
        The unique identifier of the book (default: 22077083).

    Returns
    -------
    dict
        book_id : str
        description : str
    """
    book = get_book(book_id)

    return {'book_id': str(book['book_id'].item()),
            'description': str(book['description'].item()),
            'publication_year': str(book['publication_year'].item()),
            'image_url': str(book['image_url'].item()),
            'url': str(book['url'].item()),
            'average_rating': str(book['average_rating'].item()),
            'title': str(book['title'].item()),
            'num_pages': str(book['num_pages'].item()),
            'series': str(book['series'].item())
            }
