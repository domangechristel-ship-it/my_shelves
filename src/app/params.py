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

HOST = '127.0.0.1:8000'
# HOST = 'my-shelves-image-151819310613.europe-west1.run.app'

API_URL_COUNTRY = f"http://{HOST}/country"
API_URL_BOOK_IDS_BY_COUNTRY = f"http://{HOST}/books/by-country"
API_URL_BOOKS = f"http://{HOST}/books"
API_URL_BOOK = f"http://{HOST}/read"
API_URL_BOOK_IDS_SIMILAR = f"http://{HOST}/books/similar"
