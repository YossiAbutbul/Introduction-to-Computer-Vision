"""
metrics.py - classification metrics (dog = positive class = 1).
"""

from sklearn import metrics as skm


def binary_metrics(p_dog, y, threshold=0.5):
    """p_dog: predicted P(dog) per image, y: 0 cat / 1 dog."""
    pred = (p_dog >= threshold).astype(int)
    return {
        "accuracy": skm.accuracy_score(y, pred),
        "precision": skm.precision_score(y, pred),
        "recall": skm.recall_score(y, pred),
        "f1": skm.f1_score(y, pred),
        "roc_auc": skm.roc_auc_score(y, p_dog),
        "average_precision": skm.average_precision_score(y, p_dog),
        # rows = true, columns = predicted, order cat, dog
        "confusion_matrix": skm.confusion_matrix(y, pred).tolist(),
        "n_test": int(len(y)),
    }
