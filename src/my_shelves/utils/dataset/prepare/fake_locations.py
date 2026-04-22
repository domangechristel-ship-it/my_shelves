""" Génère un dataset fictif de lieux touristiques."""

import random
import pandas as pd

from faker import Faker
import countryinfo


ENVIRONMENT = ["mountain", "beach", "forest", "city"]


class FakeLocations:
    """
    Génère un dataset fictif de lieux touristiques.
    Utile pour tester des modèles de recommandation ou d'analyse de données.
    """
    def __init__(self):
        # Seed pour reproductibilité
        random.seed(42)
        Faker.seed(42)
        self.n_rows = 1000

        self.fake = Faker(locale="en_US")

    def get_fake_row_rnd(self) -> dict:
        """
        Génère une ligne de données fictives pour un lieu touristique.

        Returns
        -------
        dict
            Un dictionnaire contenant les données d'un lieu touristique.
        """

        countries = countryinfo.filter_countries(region="Western Europe")
        countries.extend(countryinfo.filter_countries(region="Americas"))

        country = random.choice(countries)
        continent = country.region()
        provinces = country.provinces()
        if provinces:
            type_lieu = random.choice(ENVIRONMENT)
            province = random.choice(provinces)
            country_name = country.name()
            return {
                "city": self.fake.city().encode('utf-8'),
                "type": type_lieu.encode('utf-8'),
                "country": country_name.encode('utf-8'),
                "continent": continent.encode('utf-8'),
                "province": province.encode('utf-8')
            }
        return {}

    def get_fake_row(self) -> dict:
        """
        Génère une ligne de données fictives pour un lieu touristique.

        Returns
        -------
        dict
            Un dictionnaire contenant les données d'un lieu touristique.
        """
        pays = ["France", "Brésil", "Allemagne", "Japon", "Canada", "Australie"]

        continents = {
            "France": "Europe",
            "Brésil": "Amérique du Sud",
            "Allemagne": "Europe",
            "Japon": "Asie",
            "Canada": "Amérique du Nord",
            "Australie": "Océanie"
        }

        country_name = random.choice(pays)
        continent = continents[country_name]
        type_lieu = random.choice(ENVIRONMENT)
        return {
            "city": self.fake.city(),
            "type": type_lieu,
            "country": country_name,
            "continent": continent,
            "province": "province_" + self.fake.state()
        }

    def generate(self, n: int=10) -> pd.DataFrame:
        """
        Génère un DataFrame avec des lieux touristiques fictifs.

        Parameters
        ----------
        n : int, optional
            Le nombre de lignes à générer (par défaut 1000).

        Returns
        -------
        pd.DataFrame
            Un DataFrame contenant les données des lieux touristiques.
        """

        data = []
        for _ in range(n):
            row = self.get_fake_row_rnd()
            # En cas de problème avec les données aléatoires, on génère une ligne plus simple
            if not row:
                row = self.get_fake_row()
            data.append(row)
        return pd.DataFrame(data)


if __name__ == "__main__":
    generator = FakeLocations()
    df = generator.generate()
    print(df.head())
    print(df.shape)
