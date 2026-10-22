"""Model zoo. Every model takes a DataFrame of feature columns (v1-v3: the cleaned features with
category dtypes; v4: features.model_frame) and returns sales on the original scale, so they can
be compared and blended directly."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor, Ridge
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


class FourierFeatures(BaseEstimator, TransformerMixin):
    """cos/sin columns of the item-id number (v4).

    The id effect looks like a wave that repeats about every 20 numbers, so two harmonics of that
    period describe it with a handful of coefficients instead of one smoothed value per number.
    drift=True adds a copy of every column times the centred id number (and that column itself),
    so the wave's level and size can change slowly from 1 to 60.
    """

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


def _units_poisson_fourier():
    """Poisson GLM (log link) on units = sales / MRP (v4).

    Units per item: an outlet level, a per-outlet 1/MRP slope, and the id-number wave (period 20,
    two harmonics, slow drift). Units behave like counts, so Poisson weighting (rows with more
    expected units count more) fits better than Gamma's equal weighting.
    """
    return UnitsRegressor(make_pipeline(
        ColumnTransformer([
            ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
            ("outlet_inverse_mrp", OutletInverseMRP(), ["Outlet_Identifier", "Item_MRP"]),
            ("id_wave", FourierFeatures(period=20.0, harmonics=2, drift=True), ["ID_Number"]),
        ]),
        PoissonRegressor(alpha=1e-4, solver="newton-cholesky", max_iter=1000),
    ))


def _gamma_fourier_id():
    """_gamma_spline_id with the 15-knot id-number spline replaced by the two-harmonic wave (v4)."""
    return make_pipeline(
        ColumnTransformer([
            ("outlet", OneHotEncoder(handle_unknown="ignore"), ["Outlet_Identifier"]),
            ("mrp_spline", SplineTransformer(n_knots=5, degree=3), ["Item_MRP"]),
            ("id_wave", FourierFeatures(period=19.5, harmonics=2), ["ID_Number"]),
        ]),
        GammaRegressor(alpha=1e-4, max_iter=3000),
    )


PRICE_STEP = 0.6658  # every Item_Outlet_Sales value is a whole multiple of this


def item_prices(items, mrp, sales, noise=2.0):
    """Each item's hidden unit price, learned from training rows only (v5).

    In this data, sales are exactly a whole number of units times a price on the PRICE_STEP grid,
    and each row's Item_MRP is that price plus noise between -2 and 2. The price index q must
    divide every row's sales / PRICE_STEP and sit within the noise of every row's MRP; if more
    than one does, the one nearest the item's mean MRP is used.
    """
    k = np.round(np.asarray(sales, dtype=float) / PRICE_STEP).astype(np.int64)
    rows = pd.DataFrame({"item": np.asarray(items), "mrp": np.asarray(mrp, dtype=float), "k": k})
    prices = {}
    for item, g in rows.groupby("item"):
        lo = max(int(np.ceil((g["mrp"].max() - noise - 1e-6) / PRICE_STEP)), 1)
        hi = int(np.floor((g["mrp"].min() + noise + 1e-6) / PRICE_STEP))
        fits = [q for q in range(lo, hi + 1) if np.all(g["k"].to_numpy() % q == 0)]
        if fits:
            prices[item] = min(fits, key=lambda q: abs(q * PRICE_STEP - g["mrp"].mean())) * PRICE_STEP
    return prices


class TruePriceRegressor(BaseEstimator, RegressorMixin):
    """Fits `estimator` with Item_MRP replaced by each item's hidden price (v5).

    The prices come from the training rows' sales only (item_prices). Every test item appears in
    train, so test rows get their exact price; a row of an unknown item keeps its Item_MRP.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def _with_true_price(self, X):
        X = X.copy()
        X["Item_MRP"] = X["Item_Identifier"].map(self.prices_).astype(float).fillna(X["Item_MRP"])
        return X

    def fit(self, X, y):
        self.prices_ = item_prices(X["Item_Identifier"], X["Item_MRP"], y)
        self.estimator_ = clone(self.estimator).fit(self._with_true_price(X), y)
        return self

    def predict(self, X):
        return self.estimator_.predict(self._with_true_price(X))


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


def get_v3_blend_models() -> dict:
    """The two models that carry all of v3's blend weight (the other six got weight 0)."""
    return {"line_x_id_number": PerOutletLinearTimesIDNumber(shrink=20.0, window=1), "gamma_spline_id": _gamma_spline_id()}


def get_v4_models() -> dict:
    """v4's blend pool. Both read only Outlet_Identifier, Item_MRP and ID_Number."""
    return {"units_poisson_fourier": _units_poisson_fourier(), "gamma_fourier_id": _gamma_fourier_id()}


def get_v5_models() -> dict:
    """v5's blend pool: v4's two models, reading each item's true price instead of the noisy MRP."""
    return {
        "units_poisson_true_price": TruePriceRegressor(_units_poisson_fourier()),
        "gamma_fourier_true_price": TruePriceRegressor(_gamma_fourier_id()),
    }
