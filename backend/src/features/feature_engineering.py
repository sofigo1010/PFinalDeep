import pandas as pd
import numpy as np
from typing import List, Optional

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    df["day"]        = idx.day
    df["dayofweek"]  = idx.dayofweek      # 0=Mon … 6=Sun
    df["month"]      = idx.month
    df["quarter"]    = idx.quarter
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    # Interacciones
    df["dow_weekend"] = df["dayofweek"] * df["is_weekend"]
    df["day_month"]   = df["day"]       * df["month"]
    return df

def add_fourier_terms(
    df: pd.DataFrame, 
    col: str, 
    period: int, 
    K: int
) -> pd.DataFrame:
    """Polinomios de Fourier de orden K sobre la columna col."""
    t = df[col] / period * 2 * np.pi
    for k in range(1, K+1):
        df[f"{col}_sin{k}"] = np.sin(k * t)
        df[f"{col}_cos{k}"] = np.cos(k * t)
    return df

def add_lag_features(
    df: pd.DataFrame,
    lags: List[int]
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    return df

def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int]
) -> pd.DataFrame:
    df = df.copy()
    # rolling sobre sales
    for w in windows:
        df[f"roll_mean_{w}"] = df["sales"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["sales"].shift(1).rolling(w).std()
    # tendencias locales: medianas móviles largas
    df["trend_med_30"] = (
        df["sales"]
          .shift(1)
          .rolling(window=30, min_periods=1, center=True)
          .median()
    )
    df["trend_med_90"] = (
        df["sales"]
          .shift(1)
          .rolling(window=90, min_periods=1, center=True)
          .median()
    )
    return df

def encode_store_id(
    df: pd.DataFrame,
    column: str = "store_id"
) -> pd.DataFrame:
    if column in df.columns:
        df = pd.get_dummies(df, columns=[column], prefix=column)
    return df

def create_features(
    df: pd.DataFrame,
    freq: Optional[str] = None,
    lags: List[int]    = [1, 7, 14],
    windows: List[int] = [7, 14, 28, 60],  # agregamos 60 días
) -> pd.DataFrame:
    df_proc = df.copy()
    # 1) Resample
    if freq is not None:
        df_proc = df_proc.resample(freq).sum()
    # 2) Índice datetime
    df_proc.index = pd.to_datetime(df_proc.index)
    # 3) Date features + interacciones
    df_proc = add_date_features(df_proc)
    # 4) Fourier mejorados
    df_proc = add_fourier_terms(df_proc, "dayofweek", period=7,  K=3)
    df_proc = add_fourier_terms(df_proc, "month",     period=12, K=5)
    # 5) Lags
    df_proc = add_lag_features(df_proc, lags)
    # 6) Rolling + tendencias
    df_proc = add_rolling_features(df_proc, windows)
    # 7) One-hot tienda
    df_proc = encode_store_id(df_proc)
    # 8) Imputar y limpiar
    df_proc = df_proc.fillna(method="bfill").fillna(method="ffill")
    return df_proc
