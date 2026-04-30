"""
Application configuration module.

This module centralizes all global configuration values used across the
application, including API endpoints and host definition.

Purpose
-------
- Avoid hardcoding URLs in multiple files
- Ensure consistency across the application
- Simplify maintenance (single place to update endpoints)

"""
LOCAL = False

HOST = (
    "http://127.0.0.1:8000"
    if LOCAL
    else "https://my-shelves-image-151819310613.europe-west1.run.app"
)

# API_URL_BOOKS = f"{BASE_API_URL}/books"
# HOST = '127.0.0.1:8000'
# HOST = 'my-shelves-image-151819310613.europe-west1.run.app'

API_URL_COUNTRY = f"{HOST}/country"
API_URL_BOOK_IDS_BY_COUNTRY = f"{HOST}/books/by-country"
API_URL_BOOKS = f"{HOST}/books"
API_URL_BOOK = f"{HOST}/read"
API_URL_BOOK_IDS_SIMILAR = f"{HOST}/books/similar"
API_URL_SIMILAR_BOOKS = f"{HOST}/books/chat-books"
API_URL_BOOK_IDS_FILTERS = f"{HOST}/books/filter"
API_URL_READ_TITLE = f"{HOST}/read/title"
