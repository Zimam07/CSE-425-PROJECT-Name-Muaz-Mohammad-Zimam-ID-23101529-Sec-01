"""
Basic clustering evaluation metrics wrapper
"""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score


def purity_score(y_true, y_pred):
    # y_true, y_pred are 1D arrays
    contingency = {}
    for t, p in zip(y_true, y_pred):
        contingency.setdefault(p, {}).setdefault(t, 0)
        contingency[p][t] += 1
    total = 0
    for p, d in contingency.items():
        total += max(d.values())
    return total / len(y_true)


def compute_metrics(X, labels_pred, labels_true=None):
    res = {}
    if len(set(labels_pred)) > 1 and len(labels_pred) > 1:
        res['silhouette'] = silhouette_score(X, labels_pred)
        res['calinski_harabasz'] = calinski_harabasz_score(X, labels_pred)
        res['davies_bouldin'] = davies_bouldin_score(X, labels_pred)
    else:
        res['silhouette'] = None
        res['calinski_harabasz'] = None
        res['davies_bouldin'] = None

    if labels_true is not None:
        res['ari'] = adjusted_rand_score(labels_true, labels_pred)
        res['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)
        res['purity'] = purity_score(labels_true, labels_pred)
    else:
        res['ari'] = None
        res['nmi'] = None
        res['purity'] = None
    return res
