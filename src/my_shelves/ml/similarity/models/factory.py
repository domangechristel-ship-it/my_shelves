"""
Factory module for creating SimilarityModel instances.
"""

from my_shelves.ml.similarity.models.base_similarity import SimilarityModel
from my_shelves.ml.similarity.models.knn_sk import BookSimilarityModel as KnnSkModel
from my_shelves.ml.similarity.models.knn_tf import BookSimilarityModelTF as KnnTFModel
from my_shelves.ml.similarity.models.sota_tf import BookSimilaritySOTATF as SotaTFModel
from my_shelves.ml.similarity.models.sota_torch import BookSimilaritySOTA as SotaTorchModel
from my_shelves.ml.similarity.models.sota_mpnet import HybridSOTAModel


class ModelFactory:
    """
    Factory class to create instances of similarity models.
    """
    @staticmethod
    def create_model(model_name: str) -> SimilarityModel:
        """
        Creates and returns an instance of a similarity model based on its name.
        """
        if model_name == "knn_sk":
            return KnnSkModel()
        elif model_name == "knn_tf":
            return KnnTFModel()
        elif model_name == "sota_tf":
            return SotaTFModel()
        elif model_name == "sota_torch":
            return SotaTorchModel()
        elif model_name == "sota_mpnet":
            return HybridSOTAModel()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
