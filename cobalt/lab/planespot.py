# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

# This code implements the planespot algorithm to detect failure modes.
from typing import Tuple

import numpy as np
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


def gmm_with_bic(X: np.ndarray, max_components: int = 10) -> Tuple[np.ndarray, int]:
    """Implements a GMM using BIC to pick number of clusters.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        Input data array
    - max_components: int, default=10
        Maximum number of mixture components to consider.

    Returns:
    - labels: np.ndarray, shape (n_samples, )
        Cluster labels for each observation
    - optimal_n_components: int
        Number of components/clusters in the best GMM selected by BIC.
    """
    # Initialize variables for tracking best model and BIC score
    lowest_bic = np.inf
    optimal_n_components = -1
    best_model = None

    # Fit GMM models with different number of components and calculate BIC
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)

        # Update if the current model has the lowest BIC score
        if bic < lowest_bic:
            lowest_bic = bic
            optimal_n_components = n_components
            best_model = gmm

    # Get cluster labels from the best model
    labels = best_model.predict(X)

    return (labels, optimal_n_components)


def plane_spot(
    embeddings: np.ndarray,
    confidence: np.ndarray,
    predicted_label: np.ndarray,
    true_label: np.ndarray,
    w: int,
    dimensionality_reduction_algo: str = "pca",
    gmm_max_components: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements planespot algorithm.

    Parameters:
    - embeddings: np.ndarray, shape (n_samples, n_features)
        Input embeddings
    - confidence: np.ndarray, shape (n_samples, )
        Confidence scores for true class for each sample
    - predicted_label: np.ndarray, shape (n_samples, )
        Predicted labels for each sample
    - true_label: np.ndarray, shape (n_samples, )
        True labels for each sample
    - w: int
        Hyperparameter weight for confidence scores
    - dimensionality_reduction_algo: str, default = "pca"
        Dimensionality reduction algorithm to use. Supported values: {"tsne", "umap", "pca"}
    - gmm_max_components: int, default = 10
        Maximum number of mixture components (clusters) to consider for GMM

    Returns:
    - cluster_labels: np.ndarray, shape (n_samples, )
        Cluster labels for each sample
    - cluster_importance: np.ndarray, shape (number_of_clusters, )
        Cluster importance for each cluster
    """
    # Reduce embedding dimensions to 2
    if dimensionality_reduction_algo == "tsne":
        two_dim_rep = TSNE(n_components=2).fit_transform(embeddings)
    elif dimensionality_reduction_algo == "umap":
        two_dim_rep = umap.UMAP(n_components=2).fit_transform(embeddings)
    elif dimensionality_reduction_algo == "pca":
        two_dim_rep = PCA(n_components=2).fit_transform(embeddings)

    # Scale the two_dim_rep to lie between 1 and 0
    scaler = MinMaxScaler()
    two_dim_rep = scaler.fit_transform(two_dim_rep)

    # Stack the two_dim_rep and confidence scores
    R = np.hstack((two_dim_rep, (w * confidence).reshape(-1, 1)))

    # Apply GMM to find clusters
    cluster_labels, number_of_clusters = gmm_with_bic(R, gmm_max_components)

    # Find errors
    errors = np.where(np.array(true_label) == np.array(predicted_label), 0, 1)

    # Calculate cluster importance as number of errors * error rate
    cluster_importance = np.zeros(number_of_clusters)
    for i in range(number_of_clusters):
        cluster_count = np.sum(cluster_labels == i)
        cluster_error = np.sum((cluster_labels == i) & (errors == 1))
        cluster_importance[i] = cluster_error * cluster_error / cluster_count

    return (cluster_labels, cluster_importance)
