# Solutions process: step by step

Cross-validation (CV) means: split train into 5 parts, learn from 4, score on the 5th, rotate through all 5, and repeat with 3 shuffles. The average RMSE tells us how the model does on rows it has not seen. Smaller is better.

## solution_v4.py (simple, explained in full)

| Step in the code | What we wanted | What we saw | Conclusion |
|---|---|---|---|
| Read train, test and the sample file | Load the data and the required output format | 8523 train rows, 5681 test rows | The output must have `Item_Identifier`, `Outlet_Identifier`, `Item_Outlet_Sales` in the sample's row order (the script asserts this). |
| `add_id_number` (FDA15 gives 15) | Turn the end of the id into a number | A number from 1 to 60 per row | A new column that carries a little information (EDA section 7). |
| Step 1: one `LinearRegression` per outlet on `Item_MRP` | The main effect: price and store | Alone it gives a CV RMSE of about 1072 (checked while building) | This is the base model. The EDA showed a straight line inside each outlet. |
| Step 2: average actual / predicted for each id number | Find which id numbers the line over or under predicts | Using the raw ratio per number: 1067.0. Smoothing over 3 neighbouring numbers: 1062.2. Smoothing over 5: 1062.5 | The raw ratio is noisy, since each number has few rows. Smoothing over 3 neighbours works best. Total gain is about 10. |
| Step 3: multiply by 1.0278 | Match the level of the test set | One leaderboard probe showed test sales run about 2.8% above train | This number comes from leaderboard feedback, not from train. It is a small calibration. |
| `fit_and_predict` inside the CV loop | Score honestly | CV RMSE: **1062.2** | The ratio table is built from the learning part only, so there is no cheating. |
| Final fit on all of train, then predict test, `clip(lower=0)` | Make the file | `solutions/submission_v4.csv`, 5681 rows | Sales cannot be negative, so we clip at 0. |

Result: 1072 (line only) -> 1062 (+ id number). The id number helps a little, and the biggest part of the error is random.

## solution_v5.py (the submitted solution)

This is the original solution that scored **1136.80 on the public leaderboard (rank 41)**. It uses tools beyond our class (Poisson and Gamma regression, a repeating wave), so here we only follow the logic.

| Step in the code | What we wanted | What we saw | Conclusion |
|---|---|---|---|
| `model_frame` | Use only columns known for every row | Outlet, MRP and id number per row | Nothing from the test file is used to fit the models. |
| `item_prices` | `Item_MRP` is noisy (the real price plus up to 2 rupees of random error). Recover each item's real price | Every sale is exactly a whole number of units times a price on a 0.6658 grid, so each item's real price can be found from its train sales | A cleaner price than MRP. Every test item also appears in train. |
| `UnitsRegressor`, `OutletInverseMRP` | Predict units sold, then sales = units x price | Units behave like counts | Predicting units suits this data better than predicting rupees directly. |
| `FourierFeatures` | Describe the id-number effect with a wave | The effect repeats about every 20 numbers | A smoother and more compact version of v4's id correction. |
| Two models: Poisson on units, Gamma on sales | Get two different views | CV RMSE: Poisson 1059.6, Gamma 1060.6 | Both beat v4's 1062.2. |
| `blend_weights` | Combine the two models | Weights 0.655 and 0.345, blend RMSE **1059.4** | The blend is slightly better than either model. |
| Scale 1.0278 and write the CSV | Same calibration as v4 | `solutions/submission_v5.csv` | This is byte for byte the file that was submitted. |

## Comparison

| Version | CV RMSE | Public score | What was added |
|---|---|---|---|
| Average for everyone | about 1650 | n/a | n/a |
| Line per outlet | about 1072 | n/a | price and store |
| v4 (simple) | 1062.2 | not submitted | + id number |
| v5 (submitted) | 1059.4 | 1136.80 | + real price, units model, wave, blend |

The gap between v4 and v5 is only about 3 RMSE. The simple v4 keeps almost all of the gain with code we can explain. The CV scores are around 1060 and the public score is around 1137 because the public score uses a different set of rows (25% of test), so the two numbers are not directly comparable.

## How different people would read these solutions

### The statistician
- The per-outlet line is simple linear regression, fitted ten times. The only assumption we leaned on is "sales rise in a straight line with price inside one store". The EDA scatter supports it.
- The scatter fans out at higher prices, so errors are not equal everywhere. A statistician would note that as unequal variance and not trust a single average error for all rows.
- The id-number correction is a group average (ratio by id number). Each number has only a few rows, so raw averages are noisy. Smoothing over 3 neighbours trades a little bias for much less noise: 1067.0 became 1062.2.
- A gain of about 10 RMSE out of about 1070 is under 1%. They would ask whether it is bigger than the fold-to-fold wobble. A single fold's RMSE varies by about 29 (v5's standard deviation), which is much bigger than a gain of 3. Two versions scored on the same folds are compared more fairly, but a gain this small still needs that paired comparison before anyone calls it real.

### The data scientist
- Start simple and see how far it goes: the line per outlet gets most of the gain. Each extra idea adds a little.
- Look for structure in the data's own fields: `Item_MRP` is a price with a small random error added, and the item id carries a number. v5 uses both.
- Keep what can be explained. v4 can be explained in one minute, and v5 needs a paragraph on units, noise and waves. The gain from v5 is about 3 RMSE, so the choice depends on who has to explain it.
- The 1.0278 scale is a judgement call. It came from leaderboard feedback and not from train, so a careful person would say so out loud.

### The ML person
- Both scripts score themselves with repeated 5-fold cross-validation and never use the validation rows to learn (the id-number table and the item prices are built from the learning part only in v4; v5 does the same inside each fold).
- Benchmark ladder: average (about 1650) -> line per outlet (about 1072) -> + id number (1062) -> v5 blend (1059). Each step is smaller, which is a sign we are close to the noise floor.
- v5 blends two models (weights 0.655 and 0.345). Blending different views of the same data usually gives a small, safe gain.
- Cross-validation (about 1060) and the leaderboard (about 1137) differ because they use different rows. We trust the leaderboard only as one more sample of unseen data, and the final rank uses the larger private part.
- The trade-off to remember: the remaining error is mostly randomness in sales, so more complex models mostly add effort for little gain.
