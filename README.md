# BigMart Sales Prediction

Predict `Item_Outlet_Sales` for 5681 (product, store) pairs using 8523 rows with known sales. The score is RMSE (smaller is better).

## Folders

| Path | What it holds |
|---|---|
| `notebooks/01_eda.ipynb` | Look at the data, plots, Pearson correlation, a t-test, and a quick model |
| `solutions/solution_v4.py` | Version 4: a line per outlet + a correction for the item id number |
| `solutions/solution_v5.py` | Version 5: the original submitted solution (public 1136.80, rank 41) |
| `solutions/submission_v4.csv`, `submission_v5.csv` | The predictions each script writes. `submission_v5.csv` is the exact file that scored 1136.80 |
| `outputs/figures/` | The plots saved by the notebook |
| `EXPLANATION.md` | The ideas in plain words |
| `reference/` | Class notebook and notes |
| `dataset/` | `train_v9rqX0R.csv` and `test_AbJTz2l.csv` |

## Run

```
pip install -r requirements.txt
python solutions/solution_v4.py
python solutions/solution_v5.py
```

Run both from the project folder. Each prints its cross-validated RMSE and writes its CSV.

## Results

| Version | Cross-validated RMSE |
|---|---|
| Average sales for every row | about 1650 |
| One line per outlet | about 1072 |
| v4 (simple): + id-number correction | 1062.2 |
| v5 (original): complex blend with each item's exact price | 1059.4 (public 1136.80) |

`solution_v5.py` reproduces `submission_v5.csv` exactly. v5 uses models beyond class level (Poisson and Gamma regression, a sine/cosine wave over the id number), so it is explained only in outline in `EXPLANATION.md`. The simple v4 is the one to explain in detail. Older versions are in git history (commit `d8cf496`).
