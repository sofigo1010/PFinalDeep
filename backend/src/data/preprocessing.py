# src/data/preprocessing.py

import pandas as pd
import numpy as np
from src.config import LOG_TRANSFORM, FREQ

def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LOG_TRANSFORM:
        df["y"] = np.log(df["sales"] + 1)
    else:
        df["y"] = df["sales"]
    return df

def inverse_log_transform(series: pd.Series) -> pd.Series:
    if LOG_TRANSFORM:
        return np.exp(series) - 1
    return series

def preprocess_sales_data(df: pd.DataFrame, freq: str = None) -> pd.DataFrame:
    import pandas as pd, numpy as np
    from src.config import LOG_TRANSFORM, FREQ

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    fr = freq or FREQ
    s = df["sales"].resample(fr)
    df_resampled = s.sum()
    counts = s.count()
    df_resampled[counts == 0] = np.nan

    # 1) Interpolación lineal
    df_resampled = df_resampled.to_frame()
    df_resampled["sales"] = df_resampled["sales"].interpolate(
        method="linear", limit_direction="both"
    )

    # 2) Seasonal median fill (ventana 7 días centrada)
    df_resampled["sales"] = df_resampled["sales"].fillna(
        df_resampled["sales"]
          .rolling(window=7, min_periods=1, center=True)
          .median()
    )

    # 3) Winsorizar outliers 1%–99%
    low, high = (
        df_resampled["sales"].quantile(0.01),
        df_resampled["sales"].quantile(0.99)
    )
    df_resampled["sales"] = df_resampled["sales"].clip(lower=low, upper=high)

    return df_resampled
