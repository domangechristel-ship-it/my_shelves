FROM python:3.12.9-bookworm

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/my_shelves my_shelves

CMD uvicorn my_shelves.api.api:app --host 0.0.0.0 --port $PORT
