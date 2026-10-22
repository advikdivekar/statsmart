"""Model zoo. Every model takes the same input (the feature DataFrame with category dtypes)
and returns sales on the original scale, so they can be compared and blended directly."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from src import config
from src.features import CATEGORICAL


class PerOutletLinear(BaseEstimator, RegressorMixin):
    """sales = a + b * Item_MRP, fitted separately for each outlet.

    This is the strongest single idea in the data: each outlet sells roughly a fixed number of
    units per rupee of MRP. Weighted least squares (weight = 1 / MRP**power) is used because the
    noise grows with price.
    """

    def __init__(self, power=2.0):
        self.power = power

    def fit(self, X, y):
        outlet = X["Outlet_Identifier"].astype(str).to_numpy()
        mrp = X["Item_MRP"].to_numpy()
        y = np.asarray(y)
        self.coef_ = {}
        for name in np.unique(outlet):
            rows = outlet == name
            design = np.c_[np.ones(rows.sum()), mrp[rows]]
            sqrt_w = mrp[rows] ** (-self.power / 2)
            self.coef_[name] = np.linalg.lstsq(design * sqrt_w[:, None], y[rows] * sqrt_w, rcond=None)[0]
        return self

    def predict(self, X):
        outlet = X["Outlet_Identifier"].astype(str).to_numpy()
        mrp = X["Item_MRP"].to_numpy()
        return np.array([self.coef_[o][0] + self.coef_[o][1] * m for o, m in zip(outlet, mrp)])


class PerOutletLinearTimesIDNumber(BaseEstimator, RegressorMixin):
    """PerOutletLinear x (1 + effect of the item-id number).

    The effect of each ID_Number is the shrunk mean of (sales / line - 1) over its training rows
    (`shrink` pseudo-rows pull it toward 0), then smoothed over neighbouring numbers, because
    adjacent numbers behave alike (lag-1 correlation of the effects is about 0.65).
    """

    def __init__(self, shrink=20.0, window=1):
        self.shrink = shrink
        self.window = window

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.line_ = PerOutletLinear(power=2.0).fit(X, y)
        ratio = pd.Series(y / self.line_.predict(X) - 1)
        grouped = ratio.groupby(X["ID_Number"].to_numpy())
        effect = grouped.sum() / (grouped.count() + self.shrink)
        effect = effect.reindex(range(effect.index.min(), effect.index.max() + 1)).fillna(0)
        self.effect_ = effect.rolling(2 * self.window + 1, center=True, min_periods=1).mean()
        return self

    def predict(self, X):
        effect = X["ID_Number"].map(self.effect_).fillna(0).to_numpy()
        return self.line_.predict(X) * (1 + effect)


def _ridge():
    """Ridge on one-hot categories plus an Outlet x MRP interaction (= per-outlet slopes)."""
    return make_pipeline(
        _OutletMRPInteraction(),
        ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
            remainder=StandardScaler(),
        ),
        Ridge(alpha=30.0),
    )


class _OutletMRPInteraction(BaseEstimator):
    """Adds one column per outlet holding Item_MRP where the row is in that outlet, else 0."""

    def fit(self, X, y=None):
        self.outlets_ = sorted(X["Outlet_Identifier"].astype(str).unique())
        return self

    def transform(self, X):
        X = X.copy()
        outlet = X["Outlet_Identifier"].astype(str)
        for o in self.outlets_:
            X[f"MRP_x_{o}"] = np.where(outlet == o, X["Item_MRP"], 0.0)
        for c in CATEGORICAL:
            X[c] = X[c].astype(str)
        return X


def _gamma_spline():
    """Gamma GLM (log link) on outlet one-hot + a smooth spline of Item_MRP.

    The log link makes the outlet effect a multiplier on the price curve, the same structure
    as PerOutletLinear, but with a flexible curve shape. Gamma fits noise that grows with sales.
    """
    return make_pipeline(
        ColumnTransformer([
            ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
            ("mrp_spline", SplineTransformer(n_knots=5, degree=3), ["Item_MRP"]),
        ]),
        GammaRegressor(alpha=1e-4, max_iter=1000),
    )


def _gamma_spline_id():
    """_gamma_spline plus a smooth spline of the item-id number (the v3 signal)."""
    return make_pipeline(
        ColumnTransformer([
            ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
            ("mrp_spline", SplineTransformer(n_knots=5, degree=3), ["Item_MRP"]),
            ("id_number_spline", SplineTransformer(n_knots=15, degree=3), ["ID_Number"]),
        ]),
        GammaRegressor(alpha=1e-4, max_iter=3000),
    )


def _lightgbm():
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=6, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
        verbose=-1, n_jobs=-1, random_state=config.SEED,
    )


def _xgboost():
    return XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=10,
        enable_categorical=True, tree_method="hist", n_jobs=-1, random_state=config.SEED,
    )


def _catboost():
    return CatBoostRegressor(
        iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=5,
        cat_features=CATEGORICAL, verbose=0, allow_writing_files=False, thread_count=-1, random_seed=config.SEED,
    )


def get_models() -> dict:
    return {
        "per_outlet_linear": PerOutletLinear(power=2.0),
        "line_x_id_number": PerOutletLinearTimesIDNumber(shrink=20.0, window=1),
        "gamma_spline": _gamma_spline(),
        "gamma_spline_id": _gamma_spline_id(),
        "ridge_interactions": _ridge(),
        "lightgbm": _lightgbm(),
        "xgboost": _xgboost(),
        "catboost": _catboost(),
    }
