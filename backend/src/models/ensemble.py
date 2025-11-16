import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load
from xgboost import XGBRegressor

from src.models.prophet_model import load_model, _add_date_regressors
from src.config import FREQ, HORIZON_DAYS

MODEL_DIR     = Path("models")
ENSEMBLE_PATH = MODEL_DIR / "xgb_residual_adv_fixed.joblib"

# Ahora incluimos 30 en las ventanas rolling
# 1) Extiende tus ventanas rolling para incluir 365 días
WINDOWS = [7, 14, 28, 30, 60, 90, 180, 365]


def _make_features(df_fc: pd.DataFrame, resid: pd.Series = None) -> pd.DataFrame:
    df = pd.DataFrame(index=df_fc.index)

    # Componentes base de Prophet
    df["yhat"]    = df_fc["yhat"]
    df["trend"]   = df_fc["trend"]
    df["weekly"]  = df_fc["weekly"]
    df["monthly"] = df_fc["monthly"]

    # Rezagos cortos de yhat
    df["lag_1"] = df["yhat"].shift(1)
    df["lag_2"] = df["yhat"].shift(2)
    df["lag_7"] = df["yhat"].shift(7)

    # Rolling statistics sobre yhat, ahora con ventana 365 incluida
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = df["yhat"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"]  = df["yhat"].shift(1).rolling(w, min_periods=1).std()

    # Volatilidad explícita de 365 días
    df["volatility_365"] = df["roll_std_365"]

    # One-hot de día de semana y mes
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df = pd.get_dummies(df, columns=["dow","month"], drop_first=True, prefix=["dow","mon"])

    # 2) Aquí reincorporas los rezagos de residuo
    if resid is not None:
        df["resid_1"] = resid.shift(1).fillna(0)
        df["resid_2"] = resid.shift(2).fillna(0)

    # Imputación
    return df.bfill().ffill()


def train_ensemble(df_ts: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    # 1) Forecast completo de Prophet
    model  = load_model()
    future = model.make_future_dataframe(periods=HORIZON_DAYS, freq=FREQ)
    future = _add_date_regressors(future)
    fc_all = model.predict(future).set_index("ds")

    # 2) Generar y alinear features con df_ts
    feats     = _make_features(fc_all)
    feats_hist = feats.loc[df_ts.index]

    # 3) Calcular residuo histórico
    resid    = df_ts["sales"] - feats_hist["yhat"]

    # 4) Entrenar XGB simple sobre residuos
    X_train = feats_hist.drop(columns=["yhat"])
    y_train = resid.values

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)

    # 5) Guardar modelo
    dump(xgb, ENSEMBLE_PATH)


def predict_ensemble(df_ts: pd.DataFrame) -> pd.DataFrame:
    # 1) Carga XGB y Prophet
    xgb   = load(ENSEMBLE_PATH)
    model = load_model()

    # 2) Forecast completo
    future = model.make_future_dataframe(periods=HORIZON_DAYS, freq=FREQ)
    future = _add_date_regressors(future)
    fc_all = model.predict(future).set_index("ds")

    # 3) Features y horizonte de test
    feats     = _make_features(fc_all)
    feats_fut = feats.loc[df_ts.index[-HORIZON_DAYS:]]

    # 4) Predicción de residuo
    X_test     = feats_fut.drop(columns=["yhat"])
    resid_pred = xgb.predict(X_test)

    # 5) Ensamble final
    yhat_base  = feats_fut["yhat"].values
    yhat_final = yhat_base + resid_pred

    out = fc_all.iloc[-HORIZON_DAYS:][["yhat_lower","yhat_upper"]].copy()
    out["yhat"] = yhat_final
    return out.reset_index().rename(columns={"ds":"date"})


def fit_and_forecast(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper para train.py: entrena el ensemble y devuelve el forecast.
    """
    train_ensemble(df_ts)
    return predict_ensemble(df_ts)
