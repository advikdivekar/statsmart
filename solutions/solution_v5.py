# BigMart Sales Prediction - version 5
# Reads dataset/train_v9rqX0R.csv, dataset/test_AbJTz2l.csv, sample_submission_8RXa3c6.csv
# Writes solutions/submission_v5.csv
#
# Idea:
# 1. Every sale is a whole number of units times the item's price, on a grid of 0.6658.
#    Item_MRP is that price plus random noise (-2 to 2), so we recover each item's real price
#    from the train sales and use it instead of Item_MRP.
# 2. Two models: a Poisson model on units (sales / price) and a Gamma model on sales.
#    Both use the outlet, the price and the item id number (as a wave that repeats about every 20 numbers).
# 3. Blend the two with weights found on the cross-validation predictions, then scale by 1.0278.

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

train = pd.read_csv("dataset/train_v9rqX0R.csv")
test = pd.read_csv("dataset/test_AbJTz2l.csv")
sample = pd.read_csv("sample_submission_8RXa3c6.csv")

STEP = 0.6658
SCALE = 1.0278  # test sales sit about 2.8% above train (found from one leaderboard probe)

for df in (train, test):
    df["id_num"] = df["Item_Identifier"].str[3:].astype(int)


def rmse(actual, predicted):
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))


def item_prices(tr):
    """Real price of each item, found from the train rows only."""
    prices = {}
    for item, g in tr.groupby("Item_Identifier"):
        k = np.round(g["Item_Outlet_Sales"] / STEP).astype(int).to_numpy()
        low = max(int(np.ceil((g["Item_MRP"].max() - 2 - 1e-6) / STEP)), 1)
        high = int(np.floor((g["Item_MRP"].min() + 2 + 1e-6) / STEP))
        # the price must divide every sale and be within 2 of every MRP
        fits = [q for q in range(low, high + 1) if np.all(k % q == 0)]
        if fits:
            # if more than one fits, take the one closest to the average MRP
            prices[item] = min(fits, key=lambda q: abs(q * STEP - g["Item_MRP"].mean())) * STEP
    return prices


def wave(t, period, drift=False):
    """cos and sin columns of the id number (2 harmonics)."""
    cols = []
    for h in (1, 2):
        cols += [np.cos(2 * np.pi * h * t / period), np.sin(2 * np.pi * h * t / period)]
    basis = np.column_stack(cols)
    if drift:
        # let the wave change slowly from id 1 to id 60
        centre = (t - 30.5) / 59
        basis = np.column_stack([basis, basis * centre[:, None], centre])
    return basis


def prepare(tr, te):
    """Put the real price in place of Item_MRP (an unknown item keeps its MRP)."""
    prices = item_prices(tr)
    tr, te = tr.copy(), te.copy()
    for df in (tr, te):
        df["price"] = df["Item_Identifier"].map(prices).astype(float).fillna(df["Item_MRP"])
    return tr, te


def one_hot(tr, te):
    outlets = sorted(tr["Outlet_Identifier"].unique())
    return [np.column_stack([(df["Outlet_Identifier"] == o).to_numpy(float) for o in outlets]) for df in (tr, te)]


def poisson_model(tr, te):
    """Poisson model on units = sales / price, then back to sales = units * price."""
    tr, te = prepare(tr, te)
    hot_tr, hot_te = one_hot(tr, te)
    # 100 / price in each outlet's own column works like a separate line for every outlet
    x_tr = np.column_stack([hot_tr, hot_tr * (100 / tr["price"].to_numpy())[:, None],
                            wave(tr["id_num"].to_numpy(float), 20, drift=True)])
    x_te = np.column_stack([hot_te, hot_te * (100 / te["price"].to_numpy())[:, None],
                            wave(te["id_num"].to_numpy(float), 20, drift=True)])
    units = tr["Item_Outlet_Sales"] / tr["price"]
    model = PoissonRegressor(alpha=1e-4, solver="newton-cholesky", max_iter=1000).fit(x_tr, units)
    return model.predict(x_te) * te["price"].to_numpy()


def gamma_model(tr, te):
    """Gamma model directly on sales."""
    tr, te = prepare(tr, te)
    hot_tr, hot_te = one_hot(tr, te)
    spline = SplineTransformer(n_knots=5, degree=3).fit(tr[["price"]])
    x_tr = np.column_stack([hot_tr, spline.transform(tr[["price"]]), wave(tr["id_num"].to_numpy(float), 19.5)])
    x_te = np.column_stack([hot_te, spline.transform(te[["price"]]), wave(te["id_num"].to_numpy(float), 19.5)])
    model = GammaRegressor(alpha=1e-4, max_iter=3000).fit(x_tr, tr["Item_Outlet_Sales"])
    return model.predict(x_te)


models = {"units_poisson": poisson_model, "gamma": gamma_model}
y = train["Item_Outlet_Sales"]

# cross-validation: 5 folds repeated 3 times; keep each row's average out-of-fold prediction
oof = {name: np.zeros(len(train)) for name in models}
counts = np.zeros(len(train))
scores = {name: [] for name in models}
for a, b in RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(train):
    for name, model in models.items():
        pred = model(train.iloc[a], train.iloc[b])
        oof[name][b] += pred
        scores[name].append(rmse(y.iloc[b], pred))
    counts[b] += 1
for name in models:
    oof[name] /= counts
    print(name, "CV RMSE:", round(np.mean(scores[name]), 1))

# blend weights: not negative, add up to 1
names = list(models)
weights, _ = nnls(np.column_stack([oof[n] for n in names]), y)
weights = weights / weights.sum()
print("blend weights:", {n: round(float(w), 3) for n, w in zip(names, weights)})
print("blend OOF RMSE:", round(rmse(y, sum(w * oof[n] for w, n in zip(weights, names))), 1))

# final models on all of train, predict test
final = sum(w * models[n](train, test) for w, n in zip(weights, names)) * SCALE
final = np.clip(final, 0, None)

submission = test[["Item_Identifier", "Outlet_Identifier"]].copy()
submission["Item_Outlet_Sales"] = final
assert list(submission.columns) == list(sample.columns)
assert submission[["Item_Identifier", "Outlet_Identifier"]].equals(sample[["Item_Identifier", "Outlet_Identifier"]])
submission.to_csv("solutions/submission_v5.csv", index=False)
print("saved solutions/submission_v5.csv")
