"""Blend models with non-negative weights that sum to 1, fitted on out-of-fold predictions."""
import numpy as np
from scipy.optimize import nnls


def blend_weights(oof_predictions: dict, y) -> dict:
    names = list(oof_predictions)
    matrix = np.column_stack([oof_predictions[n] for n in names])
    weights, _ = nnls(matrix, np.asarray(y))
    weights = weights / weights.sum()
    return dict(zip(names, weights))


def apply_weights(predictions: dict, weights: dict):
    return sum(weights[n] * predictions[n] for n in predictions)
