"""
Configuration module for Google Cloud Platform (GCP) settings.

This module retrieves environment variables used to configure access to
Google Cloud services such as BigQuery.

Environment Variables
---------------------
GCP_PROJECT : str
    The GCP project ID used to initialize clients (e.g., BigQuery).

GCP_REGION : str
    The default region for GCP resources and services.

BQ_DATASET : str
    The BigQuery dataset name where tables are stored.

BQ_REGION : str
    The geographic location of the BigQuery dataset.

Notes
-----
These variables are typically defined in a `.env` file and loaded into
the environment before running the application.
If a variable is not set, its value will be `None`.
"""
import os

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
