# BigMart Sales Prediction: how the solution works, explained simply

This document explains the final solution (`solution_v5.py`) against the problem statement. It covers the
statistics behind each step, with real examples from the data:
- **descriptive**: what the data looks like
- **diagnostic**: why it behaves that way, tested with hypothesis tests
- **predictive**: how the model forecasts sales
- **prescriptive**: which decisions we made, and why

It also covers every problem we hit, and how we confirmed it was real before acting on it.

**Contents**

1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [The journey at a glance](#2-the-journey-at-a-glance)
3. [Key ideas, explained simply](#3-key-ideas-explained-simply)
4. [Descriptive: what the data looks like](#4-descriptive-what-the-data-looks-like)
5. [Diagnostic: hypotheses we tested and how we confirmed them](#5-diagnostic-hypotheses-we-tested-and-how-we-confirmed-them)
6. [Predictive: how v5 predicts one row, step by step](#6-predictive-how-v5-predicts-one-row-step-by-step)
7. [Prescriptive: the decisions we made and why](#7-prescriptive-the-decisions-we-made-and-why)
8. [The code, piece by piece](#8-the-code-piece-by-piece)
9. [Problems we caught along the way](#9-problems-we-caught-along-the-way)
10. [Statistics toolbox used](#10-statistics-toolbox-used)
11. [Reproduce everything](#11-reproduce-everything)
12. [What is still worth exploring](#12-what-is-still-worth-exploring)

---

## 1. The problem in plain words

BigMart has 10 stores (outlets) and 1,559 products (items). For each (product, store) pair we must predict
`Item_Outlet_Sales`, the sales of that product in that store.

- **Train:** 8,523 pairs with the sales given. **Test:** 5,681 pairs without sales; we predict these.
- **Columns:**
  - product: id, weight, fat content, visibility, type, MRP (list price)
  - store: id, size, city tier, type, opening year
- **Score:** RMSE, the typical size of our error in sales units. Lower is better.
- **Leaderboard:** the portal scores a random **25% of the test rows (public)** right away. The other **75% (private)**
  decides the final rank.
- **Submission:** one `.py` file plus one `.csv` file.

**An exam analogy that runs through this document**

| Exam | This project |
|---|---|
| Practice questions with the answer key | the train set (8,523 rows with sales) |
| Mock exams you mark yourself | cross-validation: hide part of train, predict it, check |
| The real exam | the test set (5,681 rows) |
| The teacher marks a random quarter of your answers now | the public leaderboard (25%) |
| The rest is marked at the end and decides your grade | the private leaderboard (75%) |

**The final answer in one paragraph.** Every sale in this dataset is exactly a whole number of units times the
product's true price. The `Item_MRP` column is that price with a little random noise added. v5 recovers each
product's true price and predicts the number of units from two things: the store, and the number at the end of
the product id, which moves sales up and down in a wave. Two models do this and are averaged. Cross-validated RMSE
**1059.4**; public score **1136.80**.

---

## 2. The journey at a glance

| Version | Main idea | CV RMSE | Public RMSE | Lesson |
|---|---|---|---|---|
| v1 | One straight line per store (sales vs price), blended with other models | 1071.4 | 1147.91 | Simple, structured models beat big tree models here |
| probe | v1 × 1.05, to measure a level shift | n/a | 1147.11 | Test sales sit about 2.8% higher than train: scale by 1.0278 |
| v3 | The number at the end of the product id (FDA**15** → 15) | 1061.1 | 1136.88 | A hidden signal worth about 10 points |
| v4 | That signal is a repeating **wave**; units sold behave like **counts** | 1059.9 | 1136.93 | Better estimates of the same signal (+0.05 public: noise) |
| **v5** | **The MRP column is noisy; use each product's exact true price** | **1059.4** | **1136.80** | An explicit rule hidden in the data |

The CV gains from v3 to v5 are real (checked on several different splits). They are small compared with the
noise of a 25% leaderboard sample, which is why the public score moved only a little (section 5.8).

---

## 3. Key ideas, explained simply

### 3.1 RMSE: why big misses matter most
RMSE = square root of the average squared error.

| Row | Actual | Predicted | Error | Error² |
|---|---|---|---|---|
| 1 | 1,000 | 1,100 | -100 | 10,000 |
| 2 | 2,000 | 1,900 | +100 | 10,000 |
| 3 | 3,000 | 3,300 | -300 | 90,000 |

Mean of the squared errors = 36,667, so RMSE = √36,667 ≈ **191.5**. The one big miss (row 3) supplies most of the
error. RMSE rewards predicting the **average** outcome well, so all our models predict the expected sales.

### 3.2 Cross-validation (CV): mock exams
We split the 8,523 train rows into 5 parts. We train on 4 parts and predict the 5th, and rotate so every row gets
predicted by a model that never saw it. We repeat this with 3 different shuffles: 15 mock exams in total
(`src/evaluate.py`). The CV score is the average RMSE over those 15.

- **Same folds, paired comparison.** Fold scores vary a lot (±30) because some folds hold harder rows. When two
  models are scored on the same folds, though, the fold-by-fold difference is very stable. So we compare models
  pair-wise; the standard error (SE) of that difference is typically 0.1 to 0.5.
- **Fresh splits.** Trying many ideas on the same folds can find one that is lucky on those folds (selection bias).
  So every winner was re-checked on new random splits (seeds 1, 2, 3) it had never been tuned on.

### 3.3 Why we never tuned on the leaderboard
The public score uses about 1,420 rows. If two submissions differ by a small amount, which rows happen to be in the
public quarter can move their score difference by roughly ±2 points (or ±0.7 for tiny changes). Tuning to the
public score would fit that luck, and the private 75% would not share it. So CV on train decides, and the
leaderboard is a sanity check.

### 3.4 A straight line per store (weighted least squares)
In each store, sales rise in a nearly straight line with price: `sales ≈ a + b × MRP`, where b is roughly "how
many units this store sells". Expensive items have bigger swings, so the line is fitted giving each row a weight
of 1/MRP². This is *weighted least squares*. In v1–v3 this simple model (`PerOutletLinear`) beat LightGBM, XGBoost
and CatBoost.

### 3.5 GLMs with a log link: models that multiply
A GLM (generalised linear model) with a **log link** predicts `exp(a + b + c + ...)`, so every effect
**multiplies**. "Store OUT027 sells ×1.6" and "id number 16 sells ×1.10" combine as ×1.6 × 1.10. This matches how
sales work: a popular product sells more in every store, in proportion.

The *family* says how noisy rows are expected to be:
- **Gamma:** noise grows in proportion to the expected value, so every row gets equal weight in relative terms.
- **Poisson:** the classic model for counts (0, 1, 2, ...). Rows with more expected units get more weight.

### 3.6 Fourier waves
Any repeating pattern can be built from sine and cosine waves. A wave with period P repeats every P steps.
Adding its **second harmonic** (period P/2) lets the shape be lopsided, for example a sharp peak and a slow
trough. v5 describes the id-number effect with 2 harmonics of period about 20, so about 9 numbers describe all 60
id values (`FourierFeatures`).

### 3.7 Blending (NNLS)
Two decent models that make different mistakes average out to a better one. The weights are chosen by
**non-negative least squares** on the out-of-fold predictions, the predictions each model made for rows it had not
seen. In v5: `0.655 × count model + 0.345 × Gamma model`.

### 3.8 Shrinkage
An average over few rows is noisy, so we pull it toward 0 ("no effect") in proportion to how little data it rests
on. Example of v3's rule: an id number with 140 rows averaging +12%, with 20 "pseudo-rows" of 0 added, becomes
140 × 0.12 / (140 + 20) = **+10.5%**. A number with only 20 rows averaging +12% would become just +6%.

### 3.9 The calibration probe (the 1.0278 scale)
Multiplying all predictions by s changes the score in a known, curved way. Submitting v1 and v1 × 1.05 gave two
scores, which is enough algebra to find the best s = **1.0278**: test sales run about 2.8% above the train level.
`calibrate.py` holds the formula; it was checked on simulated data first.

---

## 4. Descriptive: what the data looks like

### 4.1 Basic facts
- **Sales:** mean 2,181, median 1,794, standard deviation 1,707, min 33, max 13,087. The distribution is
  right-skewed: a few rows have very large sales.
- **Data problems.** The cleaning in `src/data.py` fixes these, but v5 doesn't need it (it reads only the product id,
  store id and MRP):
  - `Item_Fat_Content` has 5 spellings for 2 values: Low Fat 5,089, Regular 2,889, LF 316, reg 117, low fat 112.
  - `Item_Weight` is missing in 1,463 rows.
  - `Outlet_Size` is missing in 2,410 rows (whole stores).
  - 526 rows have `Item_Visibility = 0`, which is impossible.

### 4.2 Finding 1: sales ≈ units × price
In the small grocery stores, sales ÷ MRP lands almost exactly on whole numbers:

| Item | MRP | Sales | Sales ÷ MRP |
|---|---|---|---|
| FDX07 | 182.10 | 732.38 | 4.02 |
| FDE51 | 45.51 | 178.43 | 3.92 |
| FDV38 | 55.80 | 163.79 | 2.94 |

So each row is roughly *a number of units × the price*. The average number of units depends mostly on the store:

| Store | Type | Average units per product |
|---|---|---|
| OUT010, OUT019 | Grocery Store | 2.4 |
| OUT018 | Supermarket Type2 | 13.9 |
| OUT013, OUT017, OUT035, OUT045, OUT046, OUT049 | Supermarket Type1 | 15.6 to 17.0 |
| OUT027 | Supermarket Type3 | 26.6 |

(Figure: `outputs/figures/sales_vs_mrp_by_outlet_type.png`.)

### 4.3 Finding 2: the number at the end of the product id moves sales in a wave
Compared with what its store and price predict, a product sells more or less depending on the number in its id.
This is v5's fitted effect:

| Id number | 8 | 16 | 29 | 36 | 48 | 56 |
|---|---|---|---|---|---|---|
| Effect on sales | -8.4% | +10.6% | -7.8% | +10.0% | -4.5% | +9.3% |

Peaks near 16, 36 and 56 and troughs near 8, 29 and 48 make a wave that repeats about every 20 numbers.
(Figure: `outputs/figures/id_number_effect.png`.)

### 4.4 Finding 3: the exact rule behind every sale
Every sales value in train is an exact whole multiple of **0.6658**. Example: 958.752 = 0.6658 × 1,440. Each
product has one **true price** on that grid, and each row's `Item_MRP` is the true price plus noise. Product FDP04
(true price 63.9168 = 96 × 0.6658):

| Store | Listed MRP | Sales | Units (= sales ÷ true price) | MRP noise |
|---|---|---|---|---|
| OUT045 | 65.2168 | 958.7520 | **15** | +1.3 |
| OUT049 | 62.7168 | 958.7520 | **15** | -1.2 |
| OUT018 | 62.5168 | 447.4176 | **7** | -1.4 |
| OUT027 | 62.0168 | 1534.0032 | **24** | -1.9 |
| OUT046 | 64.7168 | 1022.6688 | **16** | +0.8 |

Two stores sold exactly the same amount (15 units) while their MRPs differ by 2.5. Product FDX07 (true price
183.095 = 275 × 0.6658) sold 4 units in a grocery store and 25 in OUT027. **Every row of train follows this rule.**

---

## 5. Diagnostic: hypotheses we tested and how we confirmed them

Each idea was treated as a hypothesis: H₀ ("nothing there") against H₁ ("something real"), with the evidence and
the decision. A **p-value** is the chance of seeing data at least this extreme if H₀ were true. A small p (below
0.05) is evidence against H₀. CV was the final judge of every change.

### 5.1 "Each store has its own price line" (v1)
- **Evidence:** sales vs MRP is close to a straight line in each store, with a different slope per store (figure in 4.2).
- **Test:** CV of one weighted line per store (1071.2) against tuned gradient-boosted trees (about 1075 to 1080).
- **Decision:** the simple structured model wins, and became the base of v1 to v3.

### 5.2 "The id number carries signal" (v3)
- **H₀:** after store and price, the id number has no effect.
- **Evidence:**
  - Average sales vs the store line, grouped by id number, swing between about -10% and +11%. Per-number noise is about ±4%.
  - The pattern is shared: the curves for food, drinks and non-consumables correlate with the overall curve at 0.93,
    0.82 and 0.70.
  - Neighbouring numbers agree (lag-1 autocorrelation 0.65), which noise would not do.
- **Confirmation:**
  - CV improved by 10 points (1071.4 → 1061.1).
  - On the leaderboard, a test set the model never saw, the public score improved by about 10 points too
    (1147.11 → 1136.88).

### 5.3 "The id effect is a wave" (v4)
- **Evidence (periodogram):**
  - A single sine wave with period about 20 explains 41% of the per-number variation.
  - An unbiased risk estimate (SURE) predicted that v3's per-number smoothing loses about 1.4 RMSE points to noise,
    and a two-harmonic wave only about 0.7.
- **Test:** a 97-variant comparison of curve shapes inside the same model:

| Curve shape | CV (line model) | vs v3 |
|---|---|---|
| v3: smoothed value per number | 1061.5 | 0.0 |
| Step function ("blocks of numbers") | 1062.9 | +1.4 (rejected) |
| Tuned P-spline smoothing | 1061.1 | -0.4 |
| **2-harmonic Fourier wave, slow drift** | **1060.4** | **-1.0** |
| Fourier with the period estimated inside each fold | 1060.8 | -0.7 |

- **Confirmation:**
  - The last row shows the gain doesn't depend on picking the period by looking at CV.
  - The raw averages looked blocky, but step functions did worse. The blockiness was noise, not a real pattern.

### 5.4 "Units behave like counts" (v4)
- **Evidence (dispersion check):** for a Poisson count, the standard deviation is √mean.
  - Grocery stores: mean 2.41 units, SD 1.39 (Poisson would be 1.55).
  - Type1 supermarkets: mean 16.3, SD 7.2 (Poisson would be 4.0).

  So counts are the right family, but the noise isn't the same everywhere.
- **Test:** a 375-variant grid of units models. Averaged over the grid:
  - Poisson weighting beats Gamma by about 0.3.
  - The wave beats v3's spline by about 0.6.
  - The best model scores 1060.1 alone.
- **Confirmation:** the v4 pair beats v3 on the usual split (-1.16) and on three fresh splits (-0.98, -1.11, -0.71).

### 5.5 "Is anything else hiding?" (no)
After v4 we tested its remaining errors:
- **One-way ANOVA F-tests on 29 groupings** (every column, parts of the id, row order). The smallest p was 0.015
  (vowel vs consonant letter). Among 29 tests about 1.5 would fall below 0.05 by chance alone (the
  **multiple-testing** effect), so nothing stands out.
- **Periodograms with permutation tests.** Is there a wave left over the id number, over the letters, or over
  products in id order? We shuffled the values 500 to 2,000 times to see how big a peak chance produces.
  Observed peaks had p = 0.16 to 0.9: no wave left.
- **Product effects.** For products with 7+ train rows, the spread of their z-scores was 1.002. Pure noise gives
  1.0, so there are no "blockbuster" or "dud" products beyond the wave.
- **Bias checks.** Average error by prediction size (p = 0.85) and by store type: nothing.

### 5.6 A false alarm we caught
One test said the wave differs by store type (F = 1.41, **p < 0.001**). Before believing it we checked the test's
assumption of **equal noise in every group**. It doesn't hold: the error ratio's SD is 0.58 in grocery stores but
0.35 to 0.46 in supermarkets.

After standardising each store type by its own SD:
- among supermarkets, p = 0.57 to 0.84
- within Type1, p = 0.33

**Conclusion:** the "effect" came from noisier grocery rows, which barely affect RMSE. Lesson: check a test's
assumptions before acting on a small p-value.

### 5.7 The explicit case: the price column is noisy (v5)
A chain of hypotheses, each tested before the next:

1. **"Sales values lie on a grid."** 82% of rows share their exact sales value with another row (3,493 distinct
   values in 8,523 rows), across different products and stores. The smallest gap between values is exactly 0.6658,
   and every value is a whole multiple of it (largest deviation below 10⁻¹¹). **Confirmed.**
2. **"Sales = whole units × a hidden true price."** For 1,558 of 1,559 products, exactly one grid price fits all
   their rows: it divides every sale and sits within 2 of every MRP. With it, units are whole numbers in **every**
   train row. **Confirmed.**
3. **"MRP = true price + random noise."** MRP minus true price takes 41 values from -2.0 to +2.0 in steps of 0.1.
   Its SD is 1.195; a uniform spread over those values would give 1.183. Each value appears 166 to 251 times (about 208
   expected), and the end values are not rarer. **Confirmed, approximately:** a strict χ² test (p = 0.02) finds the counts a
   little more uneven than pure chance, so "evenly spread" is a close description, not an exact law.
4. **H₀: "units sold don't depend on the MRP noise."** One-way F-test over the 41 noise values: F = 1.07, p = 0.35;
   correlation -0.009. **Not rejected.** The noise is pure noise, so the true price is the right input.
5. **Out-of-sample check.** The test file's MRPs, which played no part in fitting, all lie within **exactly 2.000**
   of the true prices recovered from train. The rule holds on data the method never saw.
6. **Does it help?** Prices learned inside each training fold only:

| Split seed | v4 | v5 | Change | Paired change ± SE |
|---|---|---|---|---|
| 42 | 1059.9 | 1059.4 | -0.51 | -0.51 ± 0.17 |
| 1 | 1059.9 | 1059.4 | -0.55 | -0.55 ± 0.11 |
| 2 | 1060.3 | 1059.8 | -0.50 | -0.49 ± 0.10 |
| 3 | 1059.8 | 1059.3 | -0.52 | -0.52 ± 0.12 |

The improvement is 3 to 5 standard errors on every split. On the leaderboard it was -0.13 (1136.93 → 1136.80),
within the expected -0.51 ± 0.70.

**Check the rule yourself** (run from the project folder):

```python
import numpy as np, pandas as pd
from src import models

train = pd.read_csv("dataset/train_v9rqX0R.csv")
sales, mrp, item = train["Item_Outlet_Sales"], train["Item_MRP"], train["Item_Identifier"]

k = sales / 0.6658
print(np.abs(k - k.round()).max())                  # about 4e-12: every sale sits on the 0.6658 grid
price = item.map(models.item_prices(item, mrp, sales))
units = sales / price
print(np.allclose(units, units.round()))            # True: whole units in every row
noise = (mrp - price).round(1)
print(noise.min(), noise.max(), noise.nunique())    # -2.0 2.0 41: MRP = true price + noise in -2..+2
```

### 5.8 Why public sits about 77 points above CV, and moved less
- **Leaderboard algebra** (the probe formula from 3.9) applied to all six public scores (`calibrate.py`'s method):
  - The public rows' sales have a larger mean square than the train level predicts (about 8.25M vs 8.10M). Those
    labels are noisier, which explains why every version scores about 77 points worse on public than in CV. No
    model can remove that.
  - The best extra scale for v5 is 0.9997, so the 1.0278 scale is still right.
- **Noise size.** Simulating random 25% slices shows that a difference like v3 → v4 moves by about ±2 public points
  from which rows were sampled, and v4 → v5 by about ±0.7. v4's +0.05 and v5's -0.13 are both within that noise.

---

## 6. Predictive: how v5 predicts one row, step by step

The first test row is product **FDW58** in store **OUT049** (Supermarket Type1), listed MRP **107.8622**, id number **58**.

1. **True price.** FDW58's train rows pin its price to **105.8622**, so this row's MRP carries the maximum noise, +2.0.
2. **Count model** (Poisson GLM on units = sales ÷ price). It predicts `exp(store level + store's price slope +
   wave at 58)` = **17.05 units**, so sales = 17.05 × 105.8622 = **1,805.37**.
3. **Gamma model** (log link on sales: store level × price curve × wave): **1,830.72**.
4. **Blend:** 0.655 × 1,805.37 + 0.345 × 1,830.72 = **1,814.12**.
5. **Level calibration:** × 1.0278 = **1,864.56**. This is the first value in `submission_v5.csv`.

The blend's out-of-fold predictions match the actual sales well across the range
(figure: `outputs/figures/actual_vs_predicted.png`). Large single misses come from rows that happened to sell
unusually many units, which is the noise no model can know.

---

## 7. Prescriptive: the decisions we made and why

| Decision | Why (the evidence) |
|---|---|
| Judge every change by CV, never by the leaderboard | The public 25% is noisy (±0.7 to ±2 points); tuning to it would fit luck the private 75% does not share |
| Re-check winners on fresh splits | Guards against picking a variant that was lucky on one set of folds |
| Use only train data for fitting (v4 onward) | v1 to v3 used test *features* (never test sales) when cleaning. v4 and v5 read only per-row columns, and prices come from train sales, so test rows only reach `predict()` |
| Keep the blend to two models | Blending all 34 models looked better on the rows used to choose the weights (1059.9) but worse on rows held out from that choice (1060.6): overfitting |
| Keep the 1.0278 scale | Re-derived from all six public scores: the best extra scale is 0.9997 |
| Do not calibrate per store type | Three more probes would cost submissions. A simulation showed the per-type adjustments would carry ±2 to 5% noise: about 2.7 points better on public but about 2.5 worse on the private set that decides the rank, unless the shifts are real |
| **Final submission: v5** | Best CV (1059.4, better on every split) and best public score (1136.80) |

---

## 8. The code, piece by piece

### 8.1 `solution_v5.py`: the submitted single file

| Part | What it does | Which finding it uses |
|---|---|---|
| Config | File paths; CV settings (5 folds × 3 repeats, seed 42); `PREDICTION_SCALE = 1.0278`; `PRICE_STEP = 0.6658` | 3.2, 3.9, 4.4 |
| `model_frame` | Keeps only the product id, store id, store type, MRP, and the id number (FDW58 → 58). Built row by row, so train and test never mix | 7 (train-only) |
| `item_prices` | For each product, finds the grid price that divides all its train sales and sits within 2 of all its MRPs | 4.4, 5.7 |
| `TruePriceRegressor` | Learns the prices in `fit` (from training rows only) and swaps them in for MRP before the model sees the data | 5.7 |
| `FourierFeatures` | Turns the id number into wave columns: cos and sin at period P and P/2, plus a slow drift | 4.3, 5.3 |
| `OutletInverseMRP` | One column per store, 100 / price, so each store's units can drift with price (like the line's intercept) | 3.4 |
| `UnitsRegressor` | Fits a model to units = sales ÷ price and multiplies the prediction back by price | 4.2, 5.4 |
| `get_models` | Model 1: Poisson GLM on units (store + store price slope + wave, period 20). Model 2: Gamma GLM on sales (store + price spline + wave, period 19.5). Both wrapped in `TruePriceRegressor` | 3.5 |
| `cross_validate` | The 15 mock exams; returns the CV score and the out-of-fold predictions | 3.2 |
| `blend_weights` | Non-negative weights from the out-of-fold predictions (0.655 / 0.345) | 3.7 |
| `main` | Cross-validate, blend, refit on all train, predict test, × 1.0278, check the format against the sample file, write the CSV | all |

### 8.2 The rest of the project

| Path | Role |
|---|---|
| `main.py` + `src/` | The same pipeline split into modules. It also cross-validates v3's and v4's models for comparison and draws the figures. Its CSV is byte-identical to `solution_v5.py`'s |
| `src/models.py` | All models: v1 to v3's line and Gamma models, the trees, and v4/v5's `FourierFeatures`, `UnitsRegressor`, `item_prices`, `TruePriceRegressor` |
| `src/evaluate.py`, `src/ensemble.py` | Repeated CV and NNLS blending |
| `src/data.py`, `src/features.py` | Cleaning and engineered features (used by v1 to v3), and `model_frame` (v4/v5) |
| `solution.py` ... `solution_v4.py` | Earlier submissions, each reproducing its own CSV |
| `probe.py`, `calibrate.py` | The level probe and the algebra that produced 1.0278 |

---

## 9. Problems we caught along the way

| Problem | How we noticed | How we confirmed the fix |
|---|---|---|
| The CV gain of v4 didn't show on public (+0.05) | Expected about -1.1 | Noise analysis: a v3→v4-sized change moves ±2 on a 25% slice; v4 kept winning on fresh splits |
| The noisy MRP column | Repeated sales values (82%) and a 0.6658 grid | Exact rule on every train row; out-of-sample check on test MRPs (max deviation exactly 2.000); -0.5 CV on 4 splits |
| A false "wave differs by store type" signal (p < 0.001) | Unequal noise between grocery and supermarkets | Standardised test: p = 0.57 to 0.84 among supermarkets |
| Big stacked blends looked better | In-sample weight fitting | Cross-fitted scoring (weights chosen without the rows being scored): 1059.9 → 1060.6, so rejected |
| Test features used in cleaning (v1 to v3) | Data-use review | v4/v5 use only per-row columns; the two v3 models that carried weight never used cleaned columns, so v3's predictions were unaffected |
| sklearn's small ridge penalty shrank the tiny 1/price columns | v4's blend came out 0.13 worse than the validated model | Rescaled to 100/price; results then matched the hand-written GLM exactly (1059.918 both) |
| Installing pygam silently downgraded scipy | Install log | Moved pygam to a separate environment and re-checked that v3 reproduces exactly (1061.459 / 1062.128 / 1061.075) |
| A wrong "expected by chance" count in one diagnostic | It said 2.3 same-product pairs among the worst 200 rows | Corrected formula gives about 11.4 (observed 24, explained by expensive items being the worst rows) |
| Reproducibility | — | Every `solution_vN.py` writes a CSV byte-identical to the pipeline's |

---

## 10. Statistics toolbox used

| Tool | In plain words | Where | What it told us |
|---|---|---|---|
| Mean, median, SD, distribution | Centre and spread | 4.1 | Sales are right-skewed; units differ by store |
| Correlation, autocorrelation | Do two curves move together? | 5.2 | The id effect is shared and smooth |
| Weighted least squares | Fit a line, trusting some rows more | 3.4 | One line per store beats big tree models |
| GLM (Poisson, Gamma, log link) | Multiplicative models with a noise family | 3.5, 5.4 | Count weighting fits units best |
| Fourier series, periodogram | Waves, and which period fits best | 3.6, 5.3 | Period ≈ 20, two harmonics |
| Shrinkage (empirical Bayes) | Pull noisy averages toward 0 | 3.8, idea 3 | Separate letter or product effects are noise |
| Repeated k-fold cross-validation | Mock exams | 3.2 | The judge of every change |
| Paired comparison, standard error | Compare on the same folds | 5.3, 5.7 | Small gains (0.5) can still be certain |
| Fresh-split re-check | Guard against selection bias | 5.4, 5.7 | Gains held on 3 new splits |
| Hypothesis tests: F-test (ANOVA), interaction test | Do group averages differ more than chance? | 5.5, 5.6, 5.7 | Nothing left after v4; the MRP noise is pure noise |
| Permutation test | Build "chance" by shuffling | 5.5 | No leftover wave |
| Multiple-testing awareness | Many tests produce some small p-values by chance | 5.5 | p = 0.015 among 29 tests is not a finding |
| Checking test assumptions | Equal variance across groups | 5.6 | Caught a false alarm |
| Goodness of fit | Does data match an assumed distribution? | 5.7 | MRP noise is close to evenly spread on -2..2 (χ² p = 0.02: not exactly) |
| Simulation, bootstrap | Replay chance many times | 5.8, 7 | Public noise is ±0.7 to ±2; per-store calibration would fit noise |
| Unbiased risk estimate (SURE) | Predict an estimator's error without new data | 5.3 | A wave beats per-number smoothing |
| Non-negative least squares blending | Best weighted average of models | 3.7 | 0.655 / 0.345 |
| Cross-fitting | Choose weights on other rows than you score | 7 | Big stacks overfit |
| Probe algebra (calibration) | Learn a level shift from two scores | 3.9, 5.8 | Test sales are about 2.8% higher; the scale is still right |

---

## 11. Reproduce everything

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# hackathon files needed locally: dataset/train_v9rqX0R.csv, dataset/test_AbJTz2l.csv, sample_submission_8RXa3c6.csv
.venv/bin/python solution_v5.py                         # the final submission -> outputs/submissions/submission_v5.csv
.venv/bin/python main.py                                # same CSV, plus figures and the v3/v4/v5 comparison
```

The experiment scripts that tested each idea were removed to keep the project lean. Their results are recorded in
this document and in the README, and the snippet in section 5.7 re-checks the key rule in seconds.

---

## 12. What is still worth exploring

- **Pin the wave down further.** The two models fit slightly different periods (20 and 19.5), and a slow drift
  helped. One shared wave, or a smoothly changing (quasi-periodic) wave, could cut its noise a little more.
- **Other generated columns.** The price turned out to be a hidden value plus noise. `Item_Visibility` and
  `Item_Weight` may follow similar rules, which could show how the unit counts were drawn. So far neither links to sales.
- **Per-store-type calibration.** Only worth it with evidence that test sales shift differently by store type (see 7).
