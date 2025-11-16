# train.py

#!/usr/bin/env python3

import os
from src.config import FREQ
from src.data.ingestion import load_and_prepare_data
from src.data.preprocessing import preprocess_sales_data
from src.models.prophet_model import tune_prophet, train_prophet
from src.models.ensemble import fit_and_forecast

def main():
    # 1) Carga y preprocesado
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)

    # 2) Hiperajuste de Prophet
    best = tune_prophet(df_ts)
    print(f"🏅 Mejor Prophet → CPS={best['cps']}, SPS={best['sps']}")

    # 3) Entrenar Prophet con esos hiperparámetros
    train_prophet(
        df_ts,
        changepoint_prior_scale=best["cps"],
        seasonality_prior_scale=best["sps"]
    )

    # 4) Forecast ensemble (Prophet + XGBoost residual)
    forecast = fit_and_forecast(df_ts)

    # 5) Guardar
    os.makedirs("models", exist_ok=True)
    forecast.to_csv("models/forecast_ensemble.csv", index=False)
    print("✅ Forecast ensemble guardado en models/forecast_ensemble.csv")

if __name__ == "__main__":
    main()