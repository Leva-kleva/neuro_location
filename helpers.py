from conf import *


def p_metric(y_true, y_pred, cutoff):
    target_pred = (y_pred > cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, target_pred).ravel()

    p = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

    return p


def count_score(y_test, pred):
    return p_metric(y_test, pred, cutoff=0.15)
