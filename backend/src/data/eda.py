# src/data/eda.py

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Any

def summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Devuelve estadísticas descriptivas de la columna 'sales' como dict.
    Útil para enviar JSON al frontend.
    """
    stats = df["sales"].describe().to_dict()
    # Asegurar tipos serializables
    return {k: float(v) for k, v in stats.items()}

def seasonal_components(
    df: pd.DataFrame,
    model: str = "additive",
    period: int = None
) -> pd.DataFrame:
    """
    Calcula descomposición de la serie (trend, seasonal, resid).
    Retorna DataFrame con columnas ['trend','seasonal','resid'].
    El frontend puede graficar cada serie.
    """
    # Inferir periodo si no se provee
    if period is None:
        inferred = pd.infer_freq(df.index)
        period = {"D":7, "W":52, "M":12}.get(inferred, None)
    decomposition = seasonal_decompose(df["sales"], model=model, period=period)
    comp_df = pd.DataFrame({
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "resid": decomposition.resid
    }, index=df.index)
    return comp_df

def get_eda_payload(
    df: pd.DataFrame,
    model: str = "additive",
    period: int = None
) -> Dict[str, Any]:
    """
    Combina stats y componentes en un payload listo para la API.
    """
    stats = summary_statistics(df)
    comps = seasonal_components(df, model, period)
    # Convertir índices y valores a listas
    payload = {
        "statistics": stats,
        "components": {
            "dates": [d.strftime("%Y-%m-%d") for d in comps.index],
            "trend": comps["trend"].fillna(method="bfill").tolist(),
            "seasonal": comps["seasonal"].tolist(),
            "residual": comps["resid"].tolist(),
        }
    }
    return payload
