# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import r2_score

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Evita divisiones por cero: ignora donde y_true == 0.
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def compute_mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.
    Scaling: MAE of naive forecast (lag-1).
    """
    n = y_true.shape[0]
    # denominador: MAE naive in-sample
    naive_errors = np.abs(y_true[1:] - y_true[:-1])
    scale = np.mean(naive_errors)
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / scale) if scale != 0 else np.nan

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))

def evaluate_forecast(
    actual: pd.Series,
    forecast: pd.DataFrame,
    date_col: str = "ds",
    yhat_col: str = "yhat"
) -> Dict[str, float]:
    """
    Retorna un diccionario con todas las métricas.
    """
    # Alinear por fechas
    fc = forecast.set_index(date_col)[yhat_col]
    aligned = pd.concat([actual, fc], axis=1, join='inner')
    aligned.columns = ["actual", "predicted"]
    y_true = aligned["actual"].values
    y_pred = aligned["predicted"].values

    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE (%)": compute_mape(y_true, y_pred),
        "sMAPE (%)": compute_smape(y_true, y_pred),
        "MASE": compute_mase(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred)
    }
