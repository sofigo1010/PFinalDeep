# backend/main.py
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import joblib
import torch
import holidays


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 2) Imports internos
from src.config import FREQ
from src.models.LSTM_model import LSTMModel
from src.models.prophet_model import _add_date_regressors, load_model as load_prophet

# --- enriquecimiento LSTM (18 features) ---
def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    promo_path = BASE_DIR / "data" / "promotions.csv"
    if promo_path.exists():
        promo = pd.read_csv(promo_path, parse_dates=["date"]).set_index("date")
        df["promo_flag"] = promo.reindex(df.index, fill_value=0)["promo_flag"]
    else:
        df["promo_flag"] = 0

    us_holidays = holidays.US()
    df["is_holiday"] = df.index.to_series().apply(lambda d: int(d in us_holidays))

    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"]   = df.index.is_month_start.astype(int)
    df["is_month_end"]     = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"]   = df.index.is_quarter_end.astype(int)

    hol_dates = np.array(sorted(pd.to_datetime(list(us_holidays.keys()))))
    df["days_to_holiday"] = df.index.to_series().apply(
        lambda d: int((hol_dates[hol_dates >= d][0] - d).days) if any(hol_dates >= d) else 0
    )

    lower, upper = df["sales"].quantile([0.01, 0.99])
    df["sales_smooth"] = df["sales"].clip(lower, upper)
    df["sales_log"]    = np.log1p(df["sales_smooth"])

    for w in (7, 14, 30):
        df[f"rm_{w}"]   = df["sales_log"].rolling(window=w, min_periods=1).mean()
        df[f"rstd_{w}"] = df["sales_log"].rolling(window=w, min_periods=1).std().fillna(0)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)

    return df

# --- generación de features XGB (~40 features) ---
WINDOWS = [7, 14, 28, 30, 60, 90, 180, 365]
def make_xgb_features(df_fc: pd.DataFrame, resid: Optional[pd.Series] = None) -> pd.DataFrame:
    df = pd.DataFrame(index=df_fc.index)
    for col in ("yhat", "trend", "weekly", "monthly"):
        df[col] = df_fc[col]
    df["lag_1"] = df["yhat"].shift(1)
    df["lag_2"] = df["yhat"].shift(2)
    df["lag_7"] = df["yhat"].shift(7)
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = df["yhat"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"]  = df["yhat"].shift(1).rolling(w, min_periods=1).std().fillna(0)
    df["volatility_365"] = df["roll_std_365"]
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df = pd.get_dummies(df, columns=["dow","month"], drop_first=True, prefix=["dow","mon"])
    if resid is not None:
        df["resid_1"] = resid.shift(1).reindex(df.index).fillna(0)
        df["resid_2"] = resid.shift(2).reindex(df.index).fillna(0)
    return df.bfill().ffill()

# 3) FastAPI + CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 4) Artefactos
ARTIFACTS    = BASE_DIR / "models"
SCALER_PATH  = ARTIFACTS / "lstm_scaler.pkl"
LSTM_STATE   = ARTIFACTS / "lstm_quantile.pth"
PROPHET_PATH = ARTIFACTS / "prophet_model.joblib"
XGB_PATH     = ARTIFACTS / "xgb_residual_adv_fixed.joblib"

# 5) Cargo modelos
lstm_scaler = joblib.load(SCALER_PATH)
lstm_model  = LSTMModel(
    input_size  = lstm_scaler.n_features_in_,
    hidden_size = 128,
    num_layers  = 2,
    dropout     = 0.2
)
lstm_model.load_state_dict(torch.load(LSTM_STATE, map_location="cpu"))
lstm_model.eval()

# Definimos el orden de las 18 features y extraemos min/max del log
FEATURES = [
    "sales_log","promo_flag","is_holiday",
    "is_month_start","is_month_end","is_quarter_start","is_quarter_end",
    "days_to_holiday","rm_7","rstd_7","rm_14","rstd_14","rm_30","rstd_30",
    "month_sin","month_cos","dow_sin","dow_cos"
]
IDX_LOG    = FEATURES.index("sales_log")
log_min    = lstm_scaler.data_min_[IDX_LOG]
log_max    = lstm_scaler.data_max_[IDX_LOG]
log_range  = log_max - log_min

prophet_model = load_prophet(PROPHET_PATH)
xgb_model     = joblib.load(XGB_PATH)
xgb_cols      = xgb_model.get_booster().feature_names

# 6) Schemas
class HorizonPoint(BaseModel):
    fecha:          str
    ventas_previas: float
    otras_vars:     float

class HorizonRequest(BaseModel):
    historical: List[HorizonPoint] = Field(..., min_items=1)
    horizon:    int                  = Field(..., gt=0)

class HorizonPrediction(BaseModel):
    fecha:    str
    lstm:     float
    ensemble: float

class HorizonResponse(BaseModel):
    predictions: List[HorizonPrediction]

# 7) Endpoint
@app.post("/predict-horizon", response_model=HorizonResponse)
async def predict_horizon(req: HorizonRequest):
    # a) Histórico
    df_hist = pd.DataFrame([p.dict() for p in req.historical])
    try:
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
    except:
        raise HTTPException(400, "Formato de fecha inválido")
    df_hist = df_hist.sort_values("fecha").reset_index(drop=True)

    # b) Serie con 'sales'
    df_full = (
        df_hist
        .set_index("fecha")[["ventas_previas"]]
        .rename(columns={"ventas_previas": "sales"})
    )
    df_full = enrich_features(df_full)

    # c) Residuo histórico
    df_prop_hist = pd.DataFrame({"ds": df_full.index})
    df_prop_hist = _add_date_regressors(df_prop_hist)
    fc_hist      = prophet_model.predict(df_prop_hist).set_index("ds")
    resid_hist   = df_full["sales"].reindex(fc_hist.index) - fc_hist["yhat"]

    SEQ_LEN = 60
    last_date = df_full.index[-1]
    preds     = []

    for step in range(req.horizon):
        next_date = last_date + pd.to_timedelta(1, unit=FREQ)

        # — LSTM —
        seq = df_full[FEATURES].iloc[-SEQ_LEN:].values
        print(f"[DEBUG] step={step} seq.shape={seq.shape} last sales={df_full['sales'].iloc[-5:].tolist()}")
        X_scaled = lstm_scaler.transform(seq)
        tensor   = torch.tensor(X_scaled[None], dtype=torch.float32)
        with torch.no_grad():
            p_lstm_scaled = lstm_model(tensor).item()  # valor en [0,1]
        # invertimos MinMaxScaler sobre el log:
        p_lstm_log = p_lstm_scaled * log_range + log_min
        p_lstm     = np.expm1(p_lstm_log)            # ventas reales
        p_lstm     = float(p_lstm) if np.isfinite(p_lstm) else 0.0
        print(f"[DEBUG] step={step} p_lstm_scaled={p_lstm_scaled:.6f} "
              f"=> log={p_lstm_log:.6f} => sales={p_lstm:.2f}")

        # — Prophet —
        df_prop = pd.DataFrame({"ds": [next_date]})
        df_prop = _add_date_regressors(df_prop)
        fc_next = prophet_model.predict(df_prop).set_index("ds")
        p_prop  = float(fc_next["yhat"].iloc[0]) if np.isfinite(fc_next["yhat"].iloc[0]) else 0.0
        print(f"[DEBUG] step={step} p_prophet yhat={p_prop:.2f}")

        # — XGB residual —
        feats   = make_xgb_features(fc_next, resid=resid_hist)
        X_resid = feats.drop(columns=["yhat"]).reindex(columns=xgb_cols, fill_value=0)
        p_xgb   = float(xgb_model.predict(X_resid)[0])
        print(f"[DEBUG] step={step} p_xgb_resid={p_xgb:.2f}")

        # ensemble
        p_ens = p_lstm + p_prop + p_xgb
        p_ens = float(p_ens) if np.isfinite(p_ens) else 0.0
        print(f"[DEBUG] step={step} p_ensemble={p_ens:.2f}")

        preds.append(HorizonPrediction(
            fecha    = next_date.strftime("%Y-%m-%d"),
            lstm     = p_lstm,
            ensemble = p_ens
        ))

        # actualizar histórico para la siguiente iteración
        df_full.loc[next_date, "sales"] = p_lstm
        df_full = enrich_features(df_full)
        resid_hist.loc[next_date] = p_lstm - p_prop
        last_date = next_date

    return HorizonResponse(predictions=preds)
