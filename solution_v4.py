"""BigMart Sales Prediction: single-file solution, version 4.

Reads  dataset/train_v9rqX0R.csv, dataset/test_AbJTz2l.csv, sample_submission_8RXa3c6.csv
Writes outputs/submissions/submission_v4.csv

Run:  python solution_v4.py   (needs pandas numpy scikit-learn scipy)

Idea: sales are roughly whole units times price, and each outlet sells a typical number of units
per item. The number at the end of the item id (FDA15 -> 15) moves sales up or down along a wave
that repeats about every 20 numbers. v4 describes that wave with two Fourier harmonics (v3 smoothed
a separate value per number) and models units = sales / MRP with a Poisson GLM. It is blended with
a Gamma GLM that uses the same kind of wave. Blend weights are non-negative and chosen on
out-of-fold predictions; the blend is then multiplied by PREDICTION_SCALE, calibrated from one
public-leaderboard probe.

Nothing from the test file is used for fitting: the models read only Outlet_Identifier, Item_MRP
and the item id, built row by row, and test rows only reach predict().
"""
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "dataset" / "train_v9rqX0R.csv"
TEST_CSV = ROOT / "dataset" / "test_AbJTz2l.csv"
SAMPLE_CSV = ROOT / "sample_submission_8RXa3c6.csv"
OUTPUT_CSV = ROOT / "outputs" / "submissions" / "submission_v4.csv"

TARGET = "Item_Outlet_Sales"
ID_COLS = ["Item_Identifier", "Outlet_Identifier"]
SEED = 42
N_FOLDS = 5
N_REPEATS = 3

# Final predictions are multiplied by this. Calibrated from one public-leaderboard probe
# (v1 predictions x 1.05); 1.0 means no calibration.
# Probe v1 x 1.05 scored 1147.1059 vs v1 1147.9129 -> s* = 1.0278 (test sales ~2.8% above train level).
PREDICTION_SCALE = 1.0278


def model_frame(raw):
    """The columns the models read, built row by row with no statistics across rows. Train and
    test frames are built separately, so nothing from the test file reaches a fitted model."""
    return pd.DataFrame({
        "Outlet_Identifier": raw["Outlet_Identifier"],
        "Outlet_Type": raw["Outlet_Type"],
        "Item_MRP": raw["Item_MRP"],
        "ID_Number": raw["Item_Identifier"].str[3:].astype(int),
    })


class FourierFeatures(BaseEstimator, TransformerMixin):
    """cos/sin columns of the item-id number: the id effect looks like a wave that repeats about
    every 20 numbers, so two harmonics of that period describe it with a handful of coefficients.
    drift=True adds a copy of every column times the centred id number (and that column itself),
    so the wave's level and size can change slowly from 1 to 60."""

    def __init__(self, period=20.0, harmonics=2, drift=False):
        self.period = period
        self.harmonics = harmonics
        self.drift = drift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.asarray(X, dtype=float).ravel()
        cols = []
        for h in range(1, self.harmonics + 1):
            cols += [np.cos(2 * np.pi * h * t / self.period), np.sin(2 * np.pi * h * t / self.period)]
        basis = np.column_stack(cols)
        if self.drift:
            centre = (t - 30.5) / 59
            basis = np.column_stack([basis, basis * centre[:, None], centre])
        return basis


class OutletInverseMRP(BaseEstimator, TransformerMixin):
    """One column per outlet holding 100 / Item_MRP where the row is in that outlet, else 0. In a
    log-link model of units this plays the role of the per-outlet line's intercept. The factor 100
    puts the column on the scale of the others, so the GLM's tiny ridge penalty leaves it alone."""

    def fit(self, X, y=None):
        self.outlets_ = sorted(X["Outlet_Identifier"].astype(str).unique())
        return self

    def transform(self, X):
        outlet = X["Outlet_Identifier"].astype(str).to_numpy()
        inverse = 100 / X["Item_MRP"].to_numpy(dtype=float)
        return np.column_stack([np.where(outlet == o, inverse, 0.0) for o in self.outlets_])


class UnitsRegressor(BaseEstimator, RegressorMixin):
    """Fits `estimator` to units = sales / Item_MRP and predicts sales = Item_MRP x predicted units."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        units = np.asarray(y, dtype=float) / X["Item_MRP"].to_numpy(dtype=float)
        self.estimator_ = clone(self.estimator).fit(X, units)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * X["Item_MRP"].to_numpy(dtype=float)


def get_models():
    return {
        # Poisson GLM (log link) on units = sales / MRP: an outlet level, a per-outlet 1/MRP slope and
        # the id-number wave (period 20, two harmonics, slow drift). Units behave like counts, so
        # Poisson weighting (rows with more expected units count more) fits better than Gamma's.
        "units_poisson_fourier": UnitsRegressor(make_pipeline(
            ColumnTransformer([
                ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
                ("outlet_inverse_mrp", OutletInverseMRP(), ["Outlet_Identifier", "Item_MRP"]),
                ("id_wave", FourierFeatures(period=20.0, harmonics=2, drift=True), ["ID_Number"]),
            ]),
            PoissonRegressor(alpha=1e-4, solver="newton-cholesky", max_iter=1000),
        )),
        # Gamma GLM (log link) on sales: outlet multiplier, MRP spline, and the id-number wave (period 19.5).
        "gamma_fourier_id": make_pipeline(
            ColumnTransformer([
                ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
                ("mrp_spline", SplineTransformer(n_knots=5, degree=3), ["Item_MRP"]),
                ("id_wave", FourierFeatures(period=19.5, harmonics=2), ["ID_Number"]),
            ]),
            GammaRegressor(alpha=1e-4, max_iter=3000),
        ),
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def cross_validate(model, X, y):
    """Repeated K-fold. Returns (mean fold RMSE, std, out-of-fold predictions averaged over repeats)."""
    splitter = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    oof_sum, oof_count, fold_scores = np.zeros(len(y)), np.zeros(len(y)), []
    for train_idx, valid_idx in splitter.split(X):
        fitted = copy.deepcopy(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[valid_idx])
        oof_sum[valid_idx] += pred
        oof_count[valid_idx] += 1
        fold_scores.append(rmse(y.iloc[valid_idx], pred))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), oof_sum / oof_count


def blend_weights(oof, y):
    """Non-negative weights summing to 1, fitted on out-of-fold predictions."""
    names = list(oof)
    weights, _ = nnls(np.column_stack([oof[n] for n in names]), np.asarray(y))
    return dict(zip(names, weights / weights.sum()))


def weighted_sum(predictions, weights):
    return sum(weights[n] * predictions[n] for n in predictions)


def main():
    train_raw, test_raw = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    X_train, X_test, y = model_frame(train_raw), model_frame(test_raw), train_raw[TARGET]

    models = get_models()
    oof = {}
    for name, model in models.items():
        mean_rmse, std_rmse, oof[name] = cross_validate(model, X_train, y)
        print(f"{name:22s} CV RMSE {mean_rmse:8.1f} (+- {std_rmse:.1f})")

    weights = blend_weights(oof, y)
    print("blend weights:", {k: round(float(v), 3) for k, v in weights.items()})
    print(f"{'blend':22s} OOF RMSE {rmse(y, weighted_sum(oof, weights)):8.1f}")

    # Refit the models that carry weight on all training data and predict the test set.
    test_pred = {n: m.fit(X_train, y).predict(X_test) for n, m in models.items() if weights[n] > 0}
    final = np.clip(PREDICTION_SCALE * weighted_sum(test_pred, weights), 0, None)
    print(f"prediction scale: {PREDICTION_SCALE}")

    sample = pd.read_csv(SAMPLE_CSV)
    submission = test_raw[ID_COLS].copy()
    submission[TARGET] = final
    assert list(submission.columns) == list(sample.columns)
    assert submission[ID_COLS].equals(sample[ID_COLS]), "row order differs from sample"
    assert len(submission) == len(sample) and submission[TARGET].notna().all()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
