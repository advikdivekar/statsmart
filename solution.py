"""BigMart Sales Prediction: single-file solution.

Reads  dataset/train_v9rqX0R.csv, dataset/test_AbJTz2l.csv, sample_submission_8RXa3c6.csv
Writes outputs/submissions/submission.csv

Run:  python solution.py      (needs pandas numpy scikit-learn scipy xgboost lightgbm catboost)

Idea: sales are almost a straight line in Item_MRP with a different slope for every outlet.
We fit one line per outlet, cross-validate a few other models, and blend them with
non-negative weights chosen on out-of-fold predictions.
"""
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "dataset" / "train_v9rqX0R.csv"
TEST_CSV = ROOT / "dataset" / "test_AbJTz2l.csv"
SAMPLE_CSV = ROOT / "sample_submission_8RXa3c6.csv"
OUTPUT_CSV = ROOT / "outputs" / "submissions" / "submission.csv"

TARGET = "Item_Outlet_Sales"
ID_COLS = ["Item_Identifier", "Outlet_Identifier"]
SEED = 42
N_FOLDS = 5
N_REPEATS = 3
DATA_YEAR = 2013

CATEGORICAL = [
    "Item_Fat_Content", "Item_Type", "Item_Category", "Outlet_Identifier",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]


# ----------------------------------------------------------------------------- data cleaning
def clean(df):
    """Fix the known data problems. Uses feature columns only (never the target)."""
    df = df.copy()

    # Fat content has 5 spellings for 2 values.
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )

    # Item_Weight: an item has one true weight, so fill from the same item in other rows.
    df["Item_Weight"] = df["Item_Weight"].fillna(df.groupby("Item_Identifier")["Item_Weight"].transform("mean"))
    df["Item_Weight"] = df["Item_Weight"].fillna(df.groupby("Item_Type")["Item_Weight"].transform("median"))

    # Outlet_Size is missing for whole outlets: use the most common size of the same Outlet_Type.
    size_by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s: s.mode().iat[0])
    df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Type"].map(size_by_type))

    # Visibility of 0 is impossible: treat as missing, use the item's mean visibility elsewhere.
    visibility = df["Item_Visibility"].replace(0, np.nan)
    df["Item_Visibility"] = visibility.fillna(visibility.groupby(df["Item_Identifier"]).transform("mean"))
    df["Item_Visibility"] = df["Item_Visibility"].fillna(visibility.mean())
    return df


# ----------------------------------------------------------------------------- features
def add_features(df):
    df = df.copy()
    df["Item_Category"] = df["Item_Identifier"].str[:2]  # FD food, DR drink, NC non-consumable
    df.loc[df["Item_Category"] == "NC", "Item_Fat_Content"] = "Non-Edible"
    df["Outlet_Age"] = DATA_YEAR - df["Outlet_Establishment_Year"]

    df["Visibility_vs_Item"] = df["Item_Visibility"] / df.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
    df["Visibility_vs_Outlet"] = df["Item_Visibility"] / df.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")
    df["MRP_vs_Type"] = df["Item_MRP"] / df.groupby("Item_Type")["Item_MRP"].transform("mean")
    df["MRP_per_Weight"] = df["Item_MRP"] / df["Item_Weight"]
    df["MRP_Band"] = pd.cut(df["Item_MRP"], bins=[0, 69, 136, 203, 400], labels=False)
    df["Item_Outlet_Count"] = df.groupby("Item_Identifier")["Outlet_Identifier"].transform("count")
    df["Outlet_Item_Count"] = df.groupby("Outlet_Identifier")["Item_Identifier"].transform("count")
    return df


def feature_columns(df):
    # Item_Identifier would just be memorised; Outlet_Identifier is kept (only 10 outlets).
    drop = ["Item_Identifier", TARGET, "Outlet_Establishment_Year"]
    return [c for c in df.columns if c not in drop]


def to_category_dtype(df):
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


# ----------------------------------------------------------------------------- models
class PerOutletLinear(BaseEstimator, RegressorMixin):
    """sales = a + b * Item_MRP, fitted separately for each outlet (weighted least squares,
    weight = 1 / MRP**power because the noise grows with price)."""

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


class OutletMRPInteraction(BaseEstimator):
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


def get_models():
    return {
        "per_outlet_linear": PerOutletLinear(power=2.0),
        "ridge_interactions": make_pipeline(
            OutletMRPInteraction(),
            ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder=StandardScaler()),
            Ridge(alpha=30.0),
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=6, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
            verbose=-1, n_jobs=-1, random_state=SEED,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10,
            enable_categorical=True, tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
        "catboost": CatBoostRegressor(
            iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=5,
            cat_features=CATEGORICAL, verbose=0, allow_writing_files=False,
            thread_count=-1, random_seed=SEED,
        ),
    }


# ----------------------------------------------------------------------------- evaluation + blending
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


# ----------------------------------------------------------------------------- main
def main():
    train_raw, test_raw = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    full = add_features(clean(pd.concat([train_raw, test_raw], ignore_index=True)))
    train, test = full.iloc[: len(train_raw)].reset_index(drop=True), full.iloc[len(train_raw):].reset_index(drop=True)
    cols = feature_columns(full)
    X_train, X_test, y = to_category_dtype(train[cols]), to_category_dtype(test[cols]), train[TARGET]

    models = get_models()
    oof = {}
    for name, model in models.items():
        mean_rmse, std_rmse, oof[name] = cross_validate(model, X_train, y)
        print(f"{name:20s} CV RMSE {mean_rmse:8.1f} (+- {std_rmse:.1f})")

    weights = blend_weights(oof, y)
    print("blend weights:", {k: round(float(v), 3) for k, v in weights.items()})
    print(f"{'blend':20s} OOF RMSE {rmse(y, weighted_sum(oof, weights)):8.1f}")

    # Refit the models that carry weight on all training data and predict the test set.
    test_pred = {n: m.fit(X_train, y).predict(X_test) for n, m in models.items() if weights[n] > 0}
    final = np.clip(weighted_sum(test_pred, weights), 0, None)

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
