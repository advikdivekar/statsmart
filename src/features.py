"""Feature engineering. Every feature here is built from other feature columns only."""
import pandas as pd

from src import config

CATEGORICAL = [
    "Item_Fat_Content", "Item_Type", "Item_Category", "Outlet_Identifier",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # FD = food, DR = drink, NC = non-consumable (first two letters of the item id).
    df["Item_Category"] = df["Item_Identifier"].str[:2]
    df.loc[df["Item_Category"] == "NC", "Item_Fat_Content"] = "Non-Edible"

    # The number at the end of the item id (FDA15 -> 15). Sales vary smoothly with it (up to about +-13%),
    # shared across FD/DR/NC and across letters: the strongest signal after price and outlet.
    df["ID_Number"] = df["Item_Identifier"].str[3:].astype(int)

    df["Outlet_Age"] = config.DATA_YEAR - df["Outlet_Establishment_Year"]

    # How visible is this item compared with its usual visibility, and with the other items in the outlet.
    df["Visibility_vs_Item"] = df["Item_Visibility"] / df.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
    df["Visibility_vs_Outlet"] = df["Item_Visibility"] / df.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")

    # Price relative to similar items, and price per unit weight.
    df["MRP_vs_Type"] = df["Item_MRP"] / df.groupby("Item_Type")["Item_MRP"].transform("mean")
    df["MRP_per_Weight"] = df["Item_MRP"] / df["Item_Weight"]

    # The MRP values form four clear price bands.
    df["MRP_Band"] = pd.cut(df["Item_MRP"], bins=[0, 69, 136, 203, 400], labels=False)

    # How many outlets stock this item, and how many items does this outlet stock.
    df["Item_Outlet_Count"] = df.groupby("Item_Identifier")["Outlet_Identifier"].transform("count")
    df["Outlet_Item_Count"] = df.groupby("Outlet_Identifier")["Item_Identifier"].transform("count")

    return df


def model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """The columns v4's and v5's models read, built row by row with no statistics across rows.
    Train and test frames are built separately, so nothing from the test file reaches a fitted model."""
    return pd.DataFrame({
        "Item_Identifier": raw["Item_Identifier"],
        "Outlet_Identifier": raw["Outlet_Identifier"],
        "Outlet_Type": raw["Outlet_Type"],
        "Item_MRP": raw["Item_MRP"],
        "ID_Number": raw["Item_Identifier"].str[3:].astype(int),
    })


def split_train_test(df: pd.DataFrame, n_train: int):
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    # Item_Identifier is dropped (too many values, would memorise); Outlet_Identifier is kept (only 10 outlets).
    drop = ["Item_Identifier", config.TARGET, "Outlet_Establishment_Year"]
    return [c for c in df.columns if c not in drop]


def to_category_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Category dtype is what LightGBM / XGBoost / CatBoost want for native categorical support."""
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """For linear models."""
    return pd.get_dummies(df, columns=CATEGORICAL, dtype=float)
