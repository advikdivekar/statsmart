# BigMart Sales Prediction

Predict `Item_Outlet_Sales` for 5681 (product, store) pairs using 8523 rows with known sales. The score is RMSE (smaller is better).

## Files

| Path | What it holds |
|---|---|
| `notebooks/01_eda.ipynb` | EDA: plots, outliers, Pearson correlation, a t-test, a quick model, and the sales = units x price pattern |
| `solutions/solution_v5.py` | The final solution: Poisson and Gamma models on each item's real price, blended |
| `solutions/submission_v5.csv` | The predictions the script writes (public score 1136.80, rank 41) |
| `outputs/figures/` | The plots saved by the notebook |

## Run

The data is not included. Download the BigMart files, put `train_v9rqX0R.csv` and `test_AbJTz2l.csv` in `dataset/`, and put `sample_submission_8RXa3c6.csv` in the project folder.

```
pip install -r requirements.txt
python solutions/solution_v5.py
```

Run from the project folder. It prints the cross-validated RMSE and writes `solutions/submission_v5.csv`.

## Result

| Model | RMSE |
|---|---|
| Average sales for every row | about 1650 |
| Final blend, cross-validated | 1059.4 |
| Public leaderboard | 1136.80 (rank 41) |
