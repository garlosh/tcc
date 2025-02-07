import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


def initialize_labels_with_kmeans(X_labeled, X_unlabeled, y_eval: np.ndarray, random_state, k=2, r=0.75):
    """
    Label propagation usando KMeans: rotula parte do conjunto não rotulado.
    """
    X_lab = X_labeled if isinstance(
        X_labeled, np.ndarray) else np.array(X_labeled)
    X_unlab = X_unlabeled if isinstance(
        X_unlabeled, np.ndarray) else np.array(X_unlabeled)

    X_all = np.vstack((X_lab, X_unlab))

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X_all)
    labels_all = kmeans.labels_

    # centróide do cluster "1"
    centroid = np.mean(X_all[labels_all == 1], axis=0)
    dist = np.linalg.norm(X_unlab - centroid, axis=1)
    if len(dist) == 0:
        return y_eval

    n_distant = int(np.ceil(r * len(X_unlab)))
    idx_sorted = np.argsort(dist)[::-1]
    idx_top = idx_sorted[:n_distant]

    # converter y_eval para array se não for
    if isinstance(y_eval, pd.Series):
        y_eval = y_eval.values

    unlabeled_ids = np.where(y_eval == -1)[0]
    chosen = unlabeled_ids[idx_top]

    # rotula como 0
    y_eval[chosen] = 0
    return y_eval
