"""Cross-validation helpers. Everything is judged by RMSE on the original sales scale."""
import copy

import numpy as np
from sklearn.model_selection import RepeatedKFold

from src import config


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def cross_validate(model, X, y, n_folds=config.N_FOLDS, n_repeats=config.N_REPEATS, seed=config.SEED):
    """Repeated K-fold. Returns (mean RMSE, std of fold RMSEs, out-of-fold predictions).

    Out-of-fold predictions are averaged over the repeats, so each training row gets a
    prediction from a model that never saw it. They are used later to pick ensemble weights.
    """
    splitter = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    oof_sum = np.zeros(len(y))
    oof_count = np.zeros(len(y))
    fold_scores = []

    for train_idx, valid_idx in splitter.split(X):
        fitted = copy.deepcopy(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[valid_idx])
        oof_sum[valid_idx] += pred
        oof_count[valid_idx] += 1
        fold_scores.append(rmse(y.iloc[valid_idx], pred))

    oof = oof_sum / oof_count
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), oof
