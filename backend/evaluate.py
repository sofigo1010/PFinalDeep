#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load as joblib_load

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from src.config import FREQ, HORIZON_DAYS
from src.data.ingestion       import load_and_prepare_data
from src.data.preprocessing   import preprocess_sales_data
from src.models.prophet_model import load_model, predict_prophet, _add_date_regressors
from src.models.ensemble      import _make_features, ENSEMBLE_PATH

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denom)

def main():
    # 1) Carga y preprocesado
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)

    # 2) In-sample forecast de Prophet
    prophet    = load_model()
    fc_hist_df = predict_prophet(prophet, periods=0, freq=FREQ)
    fc_hist    = (
        fc_hist_df.set_index("ds")["yhat"]
        .rename_axis("date")
        .reindex(df_ts.index)
        .bfill()
        .ffill()
    )

    # 3) Prepara el horizonte de evaluación
    idx    = df_ts.index[-HORIZON_DAYS:]
    y_true = df_ts["sales"].loc[idx].values
    y_p    = fc_hist.loc[idx].values

    # 4) Carga el XGB entrenado
    xgb_model = joblib_load(str(ENSEMBLE_PATH))

    # 5) Genera features sobre todo el histórico (in-sample)
    future   = prophet.make_future_dataframe(periods=0, freq=FREQ)
    future   = _add_date_regressors(future)
    fc_all   = prophet.predict(future).set_index("ds")
    feats    = _make_features(fc_all)
    feats_hist = feats.loc[df_ts.index]

    # 6) Extrae el fragmento del último horizonte
    feats_test = feats_hist.loc[idx]
    X_test     = feats_test.drop(columns=["yhat"])

    # 7) Predice residuo y ensambla
    resid_pred = xgb_model.predict(X_test)
    y_e        = feats_test["yhat"].values + resid_pred

    # 8) Métricas (elimina cualquier NaN que quede)
    mask   = ~np.isnan(y_e)
    y_true = y_true[mask]
    y_p    = y_p[mask]
    y_e    = y_e[mask]

    def metrics(y_pred, label):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true.clip(1e-3))) * 100
        smp  = smape(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"\n--- Métricas {label} ---")
        print(f"MAE:   {mae:.2f}")
        print(f"RMSE:  {rmse:.2f}")
        print(f"MAPE:  {mape:.2f}%")
        print(f"sMAPE: {smp:.2f}%")
        print(f"R²:    {r2:.3f}")

    metrics(y_p, "Prophet (in-sample)")
    metrics(y_e, "Ensemble (in-sample)")

    # 9) Diagnóstico de residuos
    residuals = y_true - y_e

    fig, axes = plt.subplots(2, 1, figsize=(8,6))
    plot_acf (residuals, lags=30, ax=axes[0], title="ACF de residuos")
    plot_pacf(residuals, lags=30, ax=axes[1], title="PACF de residuos")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.scatter(y_e, residuals, alpha=0.3)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("yhat_ensemble")
    plt.ylabel("Residuo")
    plt.title("Residuals vs. Fitted")
    plt.show()

    # 10) Guardar métricas
    os.makedirs("models", exist_ok=True)
    pd.DataFrame({
        "metric": ["MAE","RMSE","MAPE","sMAPE","R2"],
        "Prophet": [
            mean_absolute_error(y_true, y_p),
            np.sqrt(mean_squared_error(y_true, y_p)),
            np.mean(np.abs((y_true - y_p) / y_true.clip(1e-3))) * 100,
            smape(y_true, y_p),
            r2_score(y_true, y_p),
        ],
        "Ensemble": [
            mean_absolute_error(y_true, y_e),
            np.sqrt(mean_squared_error(y_true, y_e)),
            np.mean(np.abs((y_true - y_e) / y_true.clip(1e-3))) * 100,
            smape(y_true, y_e),
            r2_score(y_true, y_e),
        ]
    }).to_csv("models/metrics_comparativo.csv", index=False)
    print("\n✅ Métricas comparativas guardadas en models/metrics_comparativo.csv")

if __name__ == "__main__":
    main()
