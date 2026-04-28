"""
location.py

Feature engineering utilities for extracting and enriching geographic
information from unstructured text data.

This module provides the `Location` class, which leverages:
- Named Entity Recognition (NER) via Hugging Face Transformers
  to detect location entities in text (e.g., cities, countries, regions)
- Geocoding via OpenStreetMap (Nominatim) to resolve locations
- Country metadata via CountryInfo to retrieve region and capital coordinates

Main capabilities:
------------------
- Extract location entities from a text description
- Resolve each location to:
    - country name
    - geographic region (e.g., Europe, Asia)
    - capital latitude/longitude
- Handle ambiguous or partial locations (e.g., "Siberia", "Vermont")
- Fail gracefully when no location or match is found

Typical usage:
--------------
>>> from my_shelves.ml.features import Features
>>> features = Features()
>>> features.extract_locations("This story takes place in Italy and Brussels")

Returns:
--------
List of dictionaries with the following structure:
{
    "country": str | None,
    "region": str | None,
    "capital_latlng": list | None,
    "resolved_as": str  # indicates how the location was resolved
}

Notes:
------
- The NER model (`dslim/bert-base-NER`) is loaded once at initialization.
- Geocoding requests are rate-limited to comply with OpenStreetMap usage policies.
- Results may vary depending on the quality and ambiguity of the input text.
- External API calls (geocoding) may introduce latency.

Dependencies:
-------------
- transformers
- geopy
- countryinfo
"""
import pandas as pd
from transformers import pipeline
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from countryinfo import CountryInfo
from countryinfo.exceptions import CountryNotFoundError


class Location:
    """
    Feature engineering class for extracting and enriching locations
    from text descriptions.
    """

    def __init__(self):
        # Load NER model once
        self.location_extractor = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

        # Load geocoder once
        self.geolocator = Nominatim(user_agent="my_book_project", timeout=10)
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)

    # --------------------------------------------------
    # PUBLIC METHOD
    # --------------------------------------------------
    def extract_locations(self, description: str) -> list[dict]:
        """
        Extract and enrich locations from a text description.

        Parameters
        ----------
        description : str

        Returns
        -------
        list[dict]
        """
        ner_result = self.location_extractor(description)
        location_list = self._extract_best_locations(ner_result)

        return [self._enrich_location(loc) for loc in location_list]

    def get_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and enrich locations for each book description in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the columns:
            - book_id
            - description

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per extracted location.
        """
        df = df[["book_id", "description"]].copy()

        def safe_extract(description: str) -> list[dict]:
            if pd.isna(description) or not str(description).strip():
                return []

            try:
                return self.extract_locations(str(description))
            except Exception:
                return []

        df["locations"] = df["description"].apply(safe_extract)

        df_exploded = df.explode("locations", ignore_index=True)
        df_exploded = df_exploded[df_exploded["locations"].notna()].copy()

        if df_exploded.empty:
            return pd.DataFrame(
                columns=[
                    "book_id",
                    "country",
                    "region",
                    "capital_latlng",
                    "resolved_as",
                ]
            )

        location_cols = pd.json_normalize(df_exploded["locations"])

        df_locations = pd.concat(
            [
                df_exploded.drop(columns=["description", "locations"]).reset_index(drop=True),
                location_cols.reset_index(drop=True),
            ],
            axis=1,
        )

        return df_locations
    # --------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------
    def _extract_best_locations(self, ner_result, threshold=0.80):
        """Extract best location entities from NER output."""
        locs = [ent for ent in ner_result if ent['entity_group'] == 'LOC']

        if not locs:
            return []

        high_conf = [ent for ent in locs if ent['score'] > threshold]

        if high_conf:
            words = [ent['word'] for ent in high_conf]
        else:
            best = max(locs, key=lambda x: x['score'])
            words = [best['word']]

        words = [w.strip() for w in words]

        # remove duplicates
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        return unique_words

    def _enrich_location(self, location: str) -> dict:
        try:
            if location is None or str(location).strip() == "":
                return self.empty_result("empty")

            location = str(location).strip()

            direct = self._safe_countryinfo_country(location)
            if direct:
                return direct

            return self._safe_geocode_country(location)

        except Exception:
            return self.empty_result("error")

    def _safe_countryinfo_country(self, location: str):
        try:
            info = CountryInfo(location)
            return {
                "country": info.name(),
                "region": info.region(),
                "capital_latlng": info.capital_latlng(),
                "resolved_as": "direct_country"
            }
        except (ValueError, CountryNotFoundError):
            return None

    def _safe_geocode_country(self, location: str):
        """Fallback geocode → country."""
        try:
            geo = self.geocode(location, addressdetails=True, language="en")

            if geo is None:
                return self.empty_result("not_found")

            address = geo.raw.get("address", {})
            country = address.get("country")

            if not country:
                return self.empty_result("country_not_found")

            try:
                info = CountryInfo(country)
                region = info.region()
                capital_latlng = info.capital_latlng()
            except ValueError:
                region = None
                capital_latlng = None

            return {
                "country": country,
                "region": region,
                "capital_latlng": capital_latlng,
                "resolved_as": "geocoded"
            }

        except ValueError:
            return self.empty_result("error")

    def empty_result(self, reason: str):
        """Standard empty response."""
        return {
            "country": None,
            "region": None,
            "capital_latlng": None,
            "resolved_as": reason
        }
