from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def get_metric(targets, preds):
    try:
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))
    except :
        print('first trial do not have a pseudo label')
        auc, acc = 0, 0
    return auc, acc