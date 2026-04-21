from fastapi import FastAPI
from my_shelves.utils.bigquery import get_book
app = FastAPI()

@app.get('/')
def index():
    return {'ok': True}

@app.get('/read')
def read_book(book_id: int =22077083):

    book = get_book(book_id)

    return {'book_id': str(book['book_id'].item()),
            'description': book['description'].item()
            }
