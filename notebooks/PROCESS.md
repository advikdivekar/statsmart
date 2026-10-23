# EDA process: cell by cell

For every step of `01_eda.ipynb`: what we wanted to see, what we saw, and the conclusion. All numbers come from the executed notebook.

## 1. First look

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| `df.shape` | How big is the train data? | 8523 rows, 12 columns | Small enough to look at directly. |
| `df.head()` | What does a row look like? | One row = one item in one outlet, e.g. FDA15 in OUT049 with MRP 249.81 and sales 3735.14 | The target is `Item_Outlet_Sales`. The ids are text. |
| `df.info()` | Column types and non-null counts | 4 numeric columns, 6 text columns, 1 integer (the year). `Item_Weight` has 7060 non-null and `Outlet_Size` 6113 | Two columns have gaps. |
| `df.describe()` | Typical values and spread | Sales: mean 2181, median 1794, std 1706, so the mean sits above the median. MRP runs from 31 to 267 | Sales are skewed to the right. |
| `df.isnull().sum()` | Exactly where the missing values are | `Item_Weight` 1463, `Outlet_Size` 2410, nothing else | These are the only two columns to fill. |

## 2. Cleaning

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| `Item_Fat_Content.value_counts()` | Are the categories clean? | Five labels: Low Fat 5089, Regular 2889, LF 316, reg 117, low fat 112 | The same two things are spelled in several ways. |
| `replace({...})` | Merge the spellings | Now only Low Fat 5517 and Regular 3006 (the counts add up to 8523) | Two clean categories. |
| `fillna` mean and mode | Remove the missing values | `isnull().sum()` is 0 for every column | Data is complete. We do not use these two columns for prediction, so the fill is only for tidiness. |
| `(Item_Visibility == 0).sum()` | Are there impossible values? | 526 rows have visibility 0 | A shelf item cannot have 0 display area. We note it and do not use the column. |

## 3. One column at a time

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| Histogram of sales | Shape of the target | Most rows are low, with a long tail to the right | Skewed right. A few big sales will drive the RMSE. |
| Boxplot of sales | Are there extreme values? | Many points beyond the upper whisker | Confirmed in the next cell. |
| IQR rule on sales and MRP | Count the outliers with the class method | Sales: bounds -2566 and 6502, 186 outliers. MRP: bounds -44 and 323, 0 outliers | The sales outliers are real big sales, not mistakes, so we keep them. MRP has none. |
| Histogram of MRP | Shape of the price | Several bumps, spread from 31 to 267, no single peak | Prices come in clusters. Nothing needs fixing. |
| `Outlet_Type.value_counts()` and countplot | How many rows per store type | Supermarket Type1 5577, Grocery Store 1083, Type3 935, Type2 928 | The data is mostly Type1. |

## 4. Sales against other columns

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| Scatter, MRP vs sales | Does price relate to sales? | Sales rise with MRP, and the cloud fans out at high prices | Positive link, with a lot of scatter. |
| Boxplot, sales by outlet type | Do store types differ? | Grocery boxes sit near 0. Type3 sits highest. Types 1 and 2 are in between | Store type matters a lot. |
| `groupby('Outlet_Type')` mean | The averages behind the boxes | Grocery 340, Type2 1995, Type1 2316, Type3 3694 | Grocery sells about 7 times less than Type1. |
| `groupby('Outlet_Identifier')` mean | Do single outlets differ? | The two grocery stores OUT010 and OUT019 are about 340, and OUT027 (Type3) is 3694. The other seven are 1995 to 2439 | Each outlet has its own level, so one model per outlet makes sense. |
| Correlation heatmap | Which numeric columns link to sales? | MRP 0.57, Item_Visibility -0.13, Establishment_Year -0.05, Item_Weight 0.01 | Only MRP has a clear link. |

## 5. Pearson correlation

H0: no straight-line relationship. We reject H0 if the p-value is below 0.05.

| Column | r | p-value | Conclusion |
|---|---|---|---|
| `Item_MRP` | 0.568 | about 0 | Strong enough and clearly real. Build on this. |
| `Item_Weight` | 0.012 | 0.29 | Cannot reject H0: no evidence of a link. |
| `Item_Visibility` | -0.129 | about 0 | Weak. Grocery stores show items with larger visibility (0.105 against about 0.06) and sell less, so this mostly comes from store type. |
| `Outlet_Establishment_Year` | -0.049 | about 0 | The link is real but too weak to help. |

Note: with 8523 rows even a tiny r gets a tiny p-value. So a small p alone does not mean a useful column; r matters too.

## 6. t-test: grocery vs Supermarket Type1

| What we wanted to see | What we saw | Conclusion |
|---|---|---|
| Is the gap in average sales real or just chance? | Grocery mean 339.8, Type1 mean 2316.2, t = -42.8, p-value about 0 | Reject H0. Grocery stores really sell less, so the model must treat outlets separately. |

## 7. One line per outlet, and the id number

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| `lmplot` per outlet | Is sales vs MRP a straight line inside one outlet? | Ten panels, each with an upward line. The grocery stores (OUT010, OUT019) are nearly flat and low. OUT027 has the steepest line | A line per outlet is a good simple model. The slope differs by outlet. |
| Average sales by id number | Does the number at the end of the item id (FDA15 has 15) carry information? | The average swings between about 1740 and 2870, with no steady trend | It is bumpy, not flat, so it may help a little. Solution v4 uses it. |

## 8. Quick model

| Cell | What we wanted to see | What we saw | Conclusion |
|---|---|---|---|
| 80/20 split, RMSE of two models | Does the per-outlet line beat just guessing the average? | Average only: 1652. Line per outlet: 1015 | The line cuts the error by about 38%. |
| Actual vs predicted scatter | Where does the model go wrong? | Points follow the black line but spread out for big sales | Big sales are hard to predict, so the RMSE stays large. Much of the sales is random. |

## Overall conclusion

1. Price (`Item_MRP`) and the outlet explain most of what can be explained.
2. Weight, visibility and year add almost nothing.
3. The id number may add a small extra.

So the solutions use a line per outlet on price, plus the id number (see `solutions/PROCESS.md`).

The 1015 above is one 80/20 split. The cross-validated figure in `solutions/PROCESS.md` (1072) averages 15 splits, so it is steadier. Do not compare the two directly.

## How different people would read this data

The same tables and plots lead three kinds of people to ask different questions.

### The statistician: "Is this real, or could it be chance?"

| What they look at | How they think | Our evidence |
|---|---|---|
| Hypothesis tests | State H0 first, then see if the data can reject it. A small p-value rules out chance, not importance | Pearson on `Item_MRP`: r = 0.57, p about 0. t-test grocery vs Type1: p about 0 |
| Effect size next to p-value | With 8523 rows, even a tiny effect gets a tiny p-value, so r matters as much as p | `Outlet_Establishment_Year`: p about 0 but r = -0.05, so real but useless |
| Correlation is not cause | A link can come from a third thing hiding behind it | `Item_Visibility` has r = -0.13, but grocery stores have both bigger visibility (0.105 vs about 0.06) and lower sales |
| Distribution shape | Skewed data makes the mean a poor summary | Sales mean 2181 vs median 1794: right-skewed |
| Spread that grows with the level | Variation is not the same everywhere | The scatter fans out at high MRP: big items have bigger errors |

Their worry: finding a "pattern" that is only noise, especially when many columns are tried.

### The data scientist: "What does this mean for the business, and what do I do with it?"

| What they look at | How they think | Our evidence |
|---|---|---|
| Data quality first | Fix what is wrong before anything else | Five spellings for two fat-content labels. 526 impossible zero visibilities. 3873 missing values across two columns |
| Outliers: mistake or real? | Do not delete them without a reason | The 186 high sales are real big sales, so we keep them |
| Group comparisons | Which segment behaves differently? | Grocery stores sell about 7 times less (340 vs 2316), and OUT027 is the top store (3694) |
| Which columns are worth keeping | A column earns its place by showing a link | MRP and the outlet matter. Weight, visibility and year add almost nothing |
| Hidden structure in ids and codes | Ids often carry information | The number at the end of the item id moves average sales between about 1740 and 2870 |
| Explainability | Can I explain the model to a manager? | One line per outlet is easy to explain: "each store has its own price-to-sales rate" |

Their worry: building something nobody can explain or trust.

### The ML person: "Will it predict rows it has not seen?"

| What they look at | How they think | Our evidence |
|---|---|---|
| A baseline | Always compare with the dumbest model first | Predicting the average: RMSE 1652. Line per outlet: 1015 |
| The metric | The metric decides what matters. RMSE punishes big errors | Big sales dominate the error, so deleting outliers would hurt where it counts |
| Honest evaluation | Score on rows the model never learned from | 80/20 split here, repeated 5-fold cross-validation in the solutions |
| Leakage | Nothing from the answer or the test file may enter training | The solutions build features row by row and fit only on train |
| Noise ceiling | Some error cannot be removed | Points still spread widely around the line, so a large share of sales is random |
| Complexity vs gain | More complex is only better if the score improves enough | Complex v5 beats simple v4 by about 3 RMSE (see `solutions/PROCESS.md`) |
| Overfitting | A model that is too flexible learns noise | Averaging the id-number effect over 3 neighbours beat the raw per-number values (1062.2 vs 1067.0) |

Their worry: a model that scores well on the data it learned from and fails on new data.

### Where the three agree

All three arrive at the same plan:
1. Price and store carry the signal, so model the line per outlet.
2. Check with a test (statistician), explain it simply (data scientist), and score it on unseen rows (ML).
3. Stop adding complexity when it no longer buys much.
