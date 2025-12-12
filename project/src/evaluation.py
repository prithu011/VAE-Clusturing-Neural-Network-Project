"""
Evaluation utilities for clustering metrics: Silhouette, CH, DB, ARI, NMI, Purity.
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_metrics(features, labels, true_labels=None):
    results = {}
    if len(set(labels)) > 1:
        results['silhouette'] = silhouette_score(features, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        results['davies_bouldin'] = davies_bouldin_score(features, labels)
    else:
        results['silhouette'] = None
        results['calinski_harabasz'] = None
        results['davies_bouldin'] = None

    if true_labels is not None:
        results['ari'] = adjusted_rand_score(true_labels, labels)
        results['nmi'] = normalized_mutual_info_score(true_labels, labels)
        # purity
        results['purity'] = _purity(labels, true_labels)
    return results


def _purity(preds, truths):
    # preds: cluster labels, truths: ground truth labels
    contingency = {}
    for p, t in zip(preds, truths):
        contingency.setdefault(p, {})
        contingency[p].setdefault(t, 0)
        contingency[p][t] += 1
    total = 0
    for p in contingency:
        total += max(contingency[p].values())
    return total / len(preds)
