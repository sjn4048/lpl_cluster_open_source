import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class ClusterTrainer:
    train_x: np.ndarray
    valid_x: np.ndarray

    def __init__(self, algorithm: str = 'kmeans'):
        self.algorithm = algorithm

    def feed(self, train_x, valid_x):
        self.train_x = train_x
        self.valid_x = valid_x

    # find best k by silhouette score
    # noinspection PyUnresolvedReferences
    def find_best_k(self, kmax=15):
        sil = {}
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in tqdm(range(2, kmax + 1)):
            if self.algorithm == 'kmeans':
                kmeans = KMeans(n_clusters=k, n_init=100, random_state=42, max_iter=1000).fit(self.train_x)
                labels = kmeans.labels_
                sil[k] = silhouette_score(self.train_x, labels, metric='euclidean')
            elif self.algorithm == 'spectral':
                spectral = SpectralClustering(n_clusters=k, n_init=max(100, len(self.valid_x)), random_state=42).fit(
                    self.train_x)
                labels = spectral.labels_
                sil[k] = silhouette_score(self.train_x, labels, metric='euclidean')
            else:
                raise NotImplementedError()
        best = max(sil, key=sil.get)
        print(f'sil_scores: {sil}. Best: {best}')
        return best

    def train_and_cluster(self, n_clusters_):
        if self.algorithm == 'kmeans':
            kmeans = KMeans(n_init=100, n_clusters=n_clusters_, random_state=42, max_iter=1000)
            kmeans.fit(self.train_x)
            return kmeans.predict(self.valid_x), kmeans.cluster_centers_
        elif self.algorithm == 'spectral':
            spectral = SpectralClustering(n_clusters=n_clusters_, n_init=max(100, len(self.valid_x)))
            labels = spectral.fit_predict(self.valid_x)
            # simulate centroids
            centroids = []
            for i in range(min(labels), max(labels) + 1):
                centroids.append(self.valid_x[labels == i].mean(axis=0))
            return labels, centroids
        else:
            raise NotImplementedError()
