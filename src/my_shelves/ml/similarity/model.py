from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors


class Similarity(ABC):
    def __init__(self, X: pd.DataFrame):
        self.X = X
        self.model = None

    @abstractmethod
    def fit_model(self, data: pd.DataFrame) -> Self:
        pass

    @abstractmethod
    def get_similar(self, data, n_neighbors=5):
        pass



class SimilarityKNN(Similarity):
    def __init__(self, X: pd.DataFrame, n_neighbors=10, **kwargs):
        super().__init__(X)
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.X = X

        # self.algorithm = kwargs.get('algorithm', 'auto')
        # self.n_neighbors = kwargs.get('n_neighbors', 10)
        # self.leaf_size = kwargs.get('leaf_size', 30)
        # self.metric = kwargs.get('metric', 'minkowski')
        # self.metric_params = kwargs.get('metric_params', None)
        # self.n_jobs = kwargs.get('n_jobs', None)
        # self.p = kwargs.get('p', 2)
        # self.radius = kwargs.get('radius', 1.0)

    def fit_model(self) -> Self:
        self.model.fit(self.X)
        return self

    def get_similar(self, index:int, n_neighbors=5) -> list[int]:
        X_new = self.X.iloc[index - 1:index]
        distances, indices = self.model.kneighbors(X_new, n_neighbors=n_neighbors)
        neighbors_indices = indices[0]
        neighbors_book_ids = self.X.index[neighbors_indices]
        return indices[0, 1:]
        # return neighbors_book_ids.tolist()

    def get_similar_df(self, index:int, n_neighbors=5) ->pd.DataFrame:
        row = self.X.iloc[index:index + 1]
        X_new = row
        distances, indices = self.kneighbors(X_new, n_neighbors=n_neighbors)
        # df1.iloc[[1, 3, 5], :]
        neighbors_indices = indices[0]
        neighbors_book_ids = self.X.index[neighbors_indices]
        # print("Neighbor indices:", neighbors_indices)
        # print("Neighbor book_ids:", neighbors_book_ids)
        # print("All book_ids in original dataset:", all(bid in self.X.index for bid in neighbors_book_ids))
        return self.X.iloc[neighbors_book_ids]
