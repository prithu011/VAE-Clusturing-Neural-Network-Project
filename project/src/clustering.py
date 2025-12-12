"""
Clustering utilities: KMeans, Agglomerative, DBSCAN wrappers and visualization helpers.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA


def run_kmeans(features, n_clusters=8):
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(features)
    return labels, km


def run_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(features)
    return proj, pca


def run_agglomerative(features, n_clusters=8):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(features)
    return labels, model


def run_dbscan(features, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)
    return labels, model
