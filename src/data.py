"""Loading and cleaning. Cleaning only uses feature columns (never the target), so it is
safe to run on train and test together."""
import numpy as np
import pandas as pd

from src import config


def load_raw():
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data problems. `df` is train and test stacked together."""
    df = df.copy()

    # 1. Fat content has 5 spellings for 2 values.
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )

    # 2. Item_Weight: every item has one true weight, so fill from other rows of the same item.
    df["Item_Weight"] = df["Item_Weight"].fillna(
        df.groupby("Item_Identifier")["Item_Weight"].transform("mean")
    )
    df["Item_Weight"] = df["Item_Weight"].fillna(
        df.groupby("Item_Type")["Item_Weight"].transform("median")
    )

    # 3. Outlet_Size is missing for whole outlets, so fill with the most common size of the
    #    same Outlet_Type (all three missing outlets end up "Small").
    size_by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s: s.mode().iat[0])
    df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Type"].map(size_by_type))

    # 4. Visibility of 0 is impossible (an item always takes some shelf space): treat as missing
    #    and use the item's average visibility in the other outlets.
    visibility = df["Item_Visibility"].replace(0, np.nan)
    df["Item_Visibility"] = visibility.fillna(
        visibility.groupby(df["Item_Identifier"]).transform("mean")
    )
    df["Item_Visibility"] = df["Item_Visibility"].fillna(visibility.mean())

    return df
