# StatsMart: BigMart Sales Prediction

Predicting `Item_Outlet_Sales` for 5,681 (item, outlet) pairs across 10 BigMart stores, scored by RMSE.
The project went from public RMSE **1147.91 (rank #523)** to **1136.88 (rank #41)** in four submissions.
The jump came from finding a hidden signal in the item id, not from bigger models.

![Model comparison](outputs/figures/model_comparison.png)

---

## The problem

- **Train:** 8,523 rows with sales. **Test:** 5,681 rows without.
- Each row is one product in one store, with columns for:
  - the product: weight, fat content, visibility, type, MRP (list price)
  - the store: id, size, city tier, type, opening year
- **Metric:** RMSE on the test set. The public leaderboard uses 25% of test, and the private 75% decides the final rank.
- **Final submission:** a single `.py` code file plus a `.csv` of predictions.

---

## Submission history

| # | Submission | What changed | CV RMSE | Public RMSE | Rank |
|---|---|---|---|---|---|
| 1 | `solution.py` (v1) | Cleaning, one line per outlet, NNLS blend of 5 models | 1071.4 | 1147.91 | #523 |
| 2 | `solution_equal_blend.py` | Equal-weight blend of line, ridge and CatBoost | 1072.4 | 1148.22 | #523 |
| 3 | `probe.py` (v1 x 1.05) | One-number scale probe | n/a | 1147.11 | #417 |
| n/a | `solution_v2.py` (built, not submitted) | + Gamma GLM with spline of MRP, + 1.0278 scale | 1071.1 | about 1145.7 (estimated) | n/a |
| 4 | **`solution_v3.py` (v3)** | **+ item-id number effect** | **1061.1** | **1136.88** | **#41** |

CV is repeated 5-fold cross-validation (3 repeats) on the training set, with RMSE on the original sales scale.
The fold-to-fold standard deviation is about 30, but comparisons between models on the same folds are much tighter
than that: adding a pure-noise feature moved CV by only 0.0 to +0.5 points.

### What each submission taught us

**v1: get the structure right (1147.91).**
Sales are almost exactly a straight line in MRP, with a different slope for each outlet. A per-outlet weighted
least-squares line (CV 1071.2) beat LightGBM, XGBoost and CatBoost (about 1080). The blend ended up 90% line and 10% CatBoost.

**Equal-weight blend: no gain (1148.22).**
It ties v1, as CV predicted (1072.4 vs 1071.4). Blending models that all see the same signal doesn't add information.

**Scale probe: test sales are about 2.8% higher (1147.11).**
Train and test features are indistinguishable: adversarial validation AUC is 0.50, and every outlet is exactly 40% test.
Yet public RMSE sat about 77 points above CV, and a random public-size slice of our own CV errors reaches 1148 only 0.3% of the time.
To check for a level shift, we submitted v1 x 1.05. Scaling predictions by s gives

```
RMSE(s)^2 = RMSE(1)^2 - 2(s-1)·mean(p·(y-p)) + (s-1)^2·mean(p^2)
```

so the two public scores determine `mean(p·(y-p))` exactly, and the best scale is `s* = 1 + mean(p·(y-p)) / mean(p^2) = 1.0278`.
That is worth about 2 public points. `calibrate.py` does this calculation. It was checked on simulated data, where it recovered a known 10% shift as 1.09.

**v2: the Gamma GLM + spline idea (not submitted).**
A Gamma GLM (log link) with outlet one-hot and a spline of MRP scores CV 1071.7. It ties the per-outlet line because the
log link makes the outlet effect multiplicative, which is the same structure. Additive models without that interaction
score about 1130, which is where the big gains from GLM + spline approaches come from.

**v3: the item-id number (1136.88, #41).**
The number at the end of the item id (FDA**15** -> 15) moves sales up or down by as much as 10% to 15%, along a smooth curve.
Two v3 models use it, blended 60/40:
- the per-outlet line times a shrunk, smoothed effect for each number (CV 1061.5)
- a Gamma GLM with splines on MRP and the id number (CV 1062.1)

The CV gain of 10 points predicted a public score of about 1136; it landed at 1136.88.

---

## What the data shows (with examples)

### 1. The raw data needs cleaning

| Problem | Count in train | Fix |
|---|---|---|
| `Item_Fat_Content` has 5 spellings for 2 values | Low Fat 5089, Regular 2889, LF 316, reg 117, low fat 112 | Merge into 2; non-consumables become "Non-Edible" |
| `Item_Weight` missing | 1,463 | Same item's weight from other rows, else the median for its item type |
| `Outlet_Size` missing (whole outlets: OUT010, OUT017, OUT045) | 2,410 | Most common size for that outlet type |
| `Item_Visibility` = 0 (impossible) | 526 | Same item's mean visibility |

### 2. Sales are roughly units sold × price

In the grocery stores, sales divided by MRP lands almost on whole numbers:

| Item | MRP | Sales | Sales / MRP |
|---|---|---|---|
| FDX07 | 182.10 | 732.38 | 4.02 |
| FDE51 | 45.51 | 178.43 | 3.92 |
| FDV38 | 55.80 | 163.79 | 2.94 |
| FDM39 | 178.10 | 358.20 | 2.01 |
| FDC46 | 185.43 | 184.43 | 0.99 |
| FDW20 | 122.17 | 369.52 | 3.02 |

So each row is roughly a whole number of units at about the list price. The average number of units depends mostly on the outlet:

| Outlet | Type | Rows | Avg units (sales / MRP) | Avg sales |
|---|---|---|---|---|
| OUT010 | Grocery Store | 555 | 2.4 | 339 |
| OUT019 | Grocery Store | 528 | 2.4 | 340 |
| OUT018 | Supermarket Type2 | 928 | 13.9 | 1,996 |
| OUT045 | Supermarket Type1 | 929 | 15.6 | 2,192 |
| OUT013 | Supermarket Type1 | 932 | 16.0 | 2,299 |
| OUT046 | Supermarket Type1 | 930 | 16.2 | 2,278 |
| OUT049 | Supermarket Type1 | 930 | 16.6 | 2,348 |
| OUT017 | Supermarket Type1 | 926 | 16.7 | 2,341 |
| OUT035 | Supermarket Type1 | 930 | 17.0 | 2,439 |
| OUT027 | Supermarket Type3 | 935 | 26.6 | 3,694 |

![Sales vs MRP by outlet type](outputs/figures/sales_vs_mrp_by_outlet_type.png)

That is why one line per outlet (`sales = a + b·MRP`) is the strongest single model.

### 3. The hidden signal: the number in the item id

After the per-outlet line, how far sales land above or below it depends on the id number:

| Id number | Rows | Sales vs the outlet line |
|---|---|---|
| 1 to 5 | 716 | -2.8% |
| **6 to 10** | 716 | **-9.6%** |
| 11 to 14 | 553 | -2.3% |
| **15 to 18** | 569 | **+10.9%** |
| 19 to 26 | 1,183 | +1.8% |
| **27 to 31** | 718 | **-8.3%** |
| **32 to 39** | 1,137 | **+7.3%** |
| 40 to 60 | 2,931 | -0.2% |

**A concrete example from the same store (OUT027) at similar prices:**

| Id number | Example items | Rows | Avg MRP | Avg sales | Outlet line predicts | Difference |
|---|---|---|---|---|---|---|
| 8 | FDQ08, FDZ08, FDW08 | 17 | 134.9 | 3,053.5 | 3,582.2 | -14.8% |
| 16 | FDE16, FDF16, FDA16 | 12 | 147.7 | 4,340.1 | 3,917.2 | +10.8% |

![Id number effect](outputs/figures/id_number_effect.png)

**Why we trust it:**
- **Shared across categories:** the curve correlates 0.93 with the overall curve for food, 0.82 for drinks and 0.70 for non-consumables.
- **Shared across letters:** the A–M and N–Z halves of the alphabet agree at 0.53, about as well as their noise allows.
- **Smooth:** neighbouring numbers move together (lag-1 correlation 0.65), which is why smoothing helps.

**Why trees missed most of it:** as a raw feature, LightGBM only gains 4 points from it. A structured multiplier gains 10.

### 4. What did not help (all measured in CV)

| Idea | Result |
|---|---|
| Log or sqrt target | Worse (RMSE is on the raw scale): LightGBM 1107 (log) vs 1081 (raw) |
| 47 engineered features (ranks, group stats, interactions, frequency encodings) | Worse: LightGBM 1094 vs 1080 |
| More boosting rounds | Best at about 470 (LightGBM) or about 280 (CatBoost) rounds, then worse; 3,000 rounds gives 1091 to 1094 |
| Item-level target encoding (per item id) | No gain, or worse with weak shrinkage |
| Separate models per MRP price band | 1072.6 to 1075.4, slightly worse |
| Poisson / Tweedie GLMs, more spline knots | Ties or slightly worse |
| Capping extreme sales, item-average MRP | No change |
| Flags for originally-zero visibility or missing weight, item-type multipliers, a separate grocery curve | No change on top of v3 |
| Trees fitted on v3's remaining error | Worse the longer they train (1062.5 to 1066.6) |

---

## How the data was used

- **Training:** models are trained only on train rows and train sales, and cross-validation runs on train only.
- **Preprocessing:** cleaning and group features are computed on train and test feature columns stacked together, never test sales.
  v3's two models only use `Outlet_Identifier`, `Item_MRP` and the id number.
- **Leaderboard:** the 1.0278 scale is the one number derived from public-leaderboard feedback (the probe above).

---

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put the hackathon files in place first (not included in this repo):
#   dataset/train_v9rqX0R.csv, dataset/test_AbJTz2l.csv, sample_submission_8RXa3c6.csv
.venv/bin/python main.py           # about 3 minutes -> outputs/submissions/submission_v3.csv + figures
.venv/bin/python solution_v3.py    # the single-file version; writes the identical CSV
```

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Full pipeline: clean, features, cross-validate 8 models, NNLS blend, scale, write submission, figures |
| `solution.py`, `solution_equal_blend.py`, `solution_v2.py`, `solution_v3.py` | Single-file versions for the portal (it only accepts one `.py`) |
| `probe.py`, `calibrate.py` | Leaderboard scale probe and the maths that turns its score into a scale |
| `src/config.py` | Paths, seed, CV settings, `PREDICTION_SCALE` |
| `src/data.py` | Loading and cleaning |
| `src/features.py` | Feature engineering, including `ID_Number` |
| `src/models.py` | `PerOutletLinear`, `PerOutletLinearTimesIDNumber`, the Gamma-spline GLMs, ridge and the GBMs |
| `src/evaluate.py` | Repeated K-fold CV with out-of-fold predictions |
| `src/ensemble.py` | Non-negative least-squares blend weights |
| `src/plots.py` | Figures in `outputs/figures/` |
| `notebooks/01_eda.ipynb` | Exploratory analysis |
| `outputs/submissions/` | Every submitted CSV |

---

## More to do

v3 is the current best, but several directions remain:

1. **Understand the id-number curve.** It is periodic-looking, with peaks near 16, 36 and 56. A parametric form,
   such as a sum of bumps or a seasonal-style basis, could estimate it with less noise than per-number smoothing.
2. **Model the units directly.** Treat `sales / MRP` as a count, with a Poisson or negative-binomial GAM per outlet
   that includes the id-number curve, then convert back to sales.
3. **Other parts of the id.** Try 2-D smoothing over (letter, number), and hierarchical pooling of the curve per category.
4. **Strict train-only preprocessing.** Fit every imputation and group statistic on train only and confirm v3's
   predictions don't change.
5. **More diverse ensembles.** Try GAMs (`pygam`), multi-seed bagging of the GLMs, and stacking with a simple meta-model.
6. **Per-segment calibration.** Test whether the test-level shift differs by outlet type. This needs extra leaderboard
   probes, so weigh it against overfitting the public set.
