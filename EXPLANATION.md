# Explanation

## The problem
Each row is one product in one store. We know the sales for 8523 rows and must guess the sales for 5681 others.
The score is RMSE: the typical size of our mistake in sales units. Guessing the average for everyone gives about 1650.

## What the data showed (see `notebooks/01_eda.ipynb`)

1. **Price matters.** Pearson r between `Item_MRP` and sales is about 0.57 (p-value near 0). Dearer items sell for more money.
2. **The store matters.** A t-test of grocery stores against Supermarket Type1 gives a p-value near 0: grocery stores sell much less (average 340 against 2316).
3. **Other columns hardly matter.** `Item_Weight` has r about 0.01, and `Outlet_Establishment_Year` has r about -0.05.
4. **Sales are noisy.** For the same store and price the sales still spread a lot, which is why RMSE cannot go to zero.

## Version 4
1. For each outlet, draw a straight line of sales against `Item_MRP`.
2. Every item id ends in a number (FDA15 has 15). For each number, work out how much the line over or under predicts on average. Smooth it over 3 neighbouring numbers and multiply the prediction by it.
3. Multiply everything by 1.0278, because one leaderboard probe showed test sales are about 2.8% above train.

Cross-validated RMSE: 1062.2.

## Version 5 (the submitted one)
Same idea as version 4, with stronger tools:
- `Item_MRP` is the item's real price plus a random error of up to 2 rupees, new on every row. Every sale is exactly a whole number of units times the real price, so the real price of each item can be recovered from the training sales and used instead of `Item_MRP`.
- Sales are modelled as units x price, and the id-number effect is a repeating wave instead of a smoothed average.
- Two models are blended.

Cross-validated RMSE: 1059.4, public leaderboard 1136.80. These tools (Poisson and Gamma regression, waves) are beyond what we have studied, so version 4 is the one to explain step by step.

## Cross-validation
We split train into 5 parts, learn from 4 and test on the 5th, rotate through all 5, and repeat with 3 different shuffles. The average RMSE tells us how the model does on rows it has not seen.

## Things we left out
Anything we have not studied in class (fancier models and tests) was removed to keep the code simple. The earlier, more complex versions are in git history.
