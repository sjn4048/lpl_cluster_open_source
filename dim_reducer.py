import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List


class DimReducer:
    def __init__(self):
        self.model = PCA(2, random_state=42)

    def fit(self, data):
        self.model.fit(data)

    def transform(self, data):
        if len(data) == 0:
            return []
        return self.model.transform(data)


class TSNEDimReducer:
    def __init__(self, perplexity: int = 40, n_iter: int = 10000, n_components: int = 2):
        self.model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)

    def fit_transform(self, data: np.ndarray | List[np.ndarray]) -> np.ndarray | List[np.ndarray]:
        if isinstance(data, list):
            res = self.model.fit_transform(np.concatenate(data))
            split_shapes = []
            cur = 0
            for d in data:
                cur += len(d)
                split_shapes.append(cur)
            return np.split(res, split_shapes[:-1])
        return self.model.fit_transform(data)
