import os
import numpy as np
import pandas as pd
import joblib
import holidays
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.data.ingestion       import load_and_prepare_data
from src.data.preprocessing   import preprocess_sales_data
from src.models.LSTM_model    import LSTMModel
from src.models.ensemble      import ENSEMBLE_PATH, _make_features
from src.models.prophet_model import load_model, _add_date_regressors
from src.config               import FREQ, HORIZON_DAYS, LOG_TRANSFORM

# --- enriquecimiento LSTM (18 features) ---
def enrich_features(df):
    df = df.copy()
    # promo_flag
    promo_path = "data/promotions.csv"
    if os.path.exists(promo_path):
        promo = pd.read_csv(promo_path, parse_dates=["date"]).set_index("date")
        df["promo_flag"] = promo.reindex(df.index, fill_value=0)["promo_flag"]
    else:
        df["promo_flag"] = 0

    # festivos USA
    us_holidays = holidays.US()
    df["is_holiday"] = df.index.to_series().apply(lambda d: 1 if d in us_holidays else 0)

    # calendario
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"]   = df.index.is_month_start.astype(int)
    df["is_month_end"]     = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"]   = df.index.is_quarter_end.astype(int)

    # días hasta próximo festivo
    hol_dates = np.array(sorted(pd.to_datetime(list(us_holidays.keys()))))
    df["days_to_holiday"] = df.index.to_series().apply(
        lambda d: (hol_dates[hol_dates >= d][0] - d).days if any(hol_dates >= d) else 0
    )

    # log + rodantes + cíclicas
    lower, upper = df["sales"].quantile([0.01, 0.99])
    df["sales_smooth"] = df["sales"].clip(lower, upper)
    df["sales_log"]    = np.log1p(df["sales_smooth"])
    for w in [7, 14, 30]:
        df[f"rm_{w}"]   = df["sales_log"].rolling(window=w, min_periods=1).mean()
        df[f"rstd_{w}"] = df["sales_log"].rolling(window=w, min_periods=1).std().fillna(0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)

    return df

# crea secuencias deslizantes
def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)

def main():
    # 1) Carga & preprocesado
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)  # index=date, col "sales"

    # 2) Enriquecer
    df_feat = enrich_features(df_ts.copy())

    # 3) Preparar LSTM
    features   = [
        "sales_log","promo_flag","is_holiday",
        "is_month_start","is_month_end",
        "is_quarter_start","is_quarter_end",
        "days_to_holiday",
        "rm_7","rstd_7","rm_14","rstd_14","rm_30","rstd_30",
        "month_sin","month_cos","dow_sin","dow_cos"
    ]
    scaler      = joblib.load("models/lstm_scaler.pkl")
    data_scaled = scaler.transform(df_feat[features].values)

    SEQ_LEN    = 60
    target_idx = features.index("sales_log")
    X_all, y_scaled = create_sequences(data_scaled, SEQ_LEN, target_idx)

    # tomamos el último HORIZON_DAYS como "test in-sample"
    X_test       = X_all[-HORIZON_DAYS:]
    y_scaled_test = y_scaled[-HORIZON_DAYS:]
    dates        = df_feat.index[-HORIZON_DAYS:]

    # 4) Carga Quantile LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMModel(
        input_size = len(features),
        hidden_size=128,
        num_layers =2,
        dropout    =0.2
    ).to(device)
    lstm_model.load_state_dict(torch.load("models/lstm_quantile.pth", map_location=device))
    lstm_model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                      torch.zeros(len(X_test))),
        batch_size=32,
        shuffle=False
    )

    # 5) Predecir LSTM (escala -> desescalar -> expm1)
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = lstm_model(xb).cpu().numpy().flatten()
            preds_scaled.extend(out)
    preds_scaled = np.array(preds_scaled)

    # Desescalar predicciones (solo columna sales_log)
    tmp_pred = np.zeros((len(preds_scaled), len(features)))
    tmp_pred[:, target_idx] = preds_scaled
    inv_pred = scaler.inverse_transform(tmp_pred)
    sales_log_pred = inv_pred[:, target_idx]
    y_pred_lstm = np.expm1(sales_log_pred)

    # Desescalar truth
    tmp_true = np.zeros_like(tmp_pred)
    tmp_true[:, target_idx] = y_scaled_test
    inv_true = scaler.inverse_transform(tmp_true)
    sales_log_true = inv_true[:, target_idx]
    y_true = np.expm1(sales_log_true)

    # 6) Ensemble in-sample en la cola
    from joblib import load as joblib_load
    xgb   = joblib_load(ENSEMBLE_PATH)
    model = load_model()

    # Forecast completo Prophet (historia + futuro) pero usamos sólo historia
    future = model.make_future_dataframe(periods=HORIZON_DAYS, freq=FREQ)
    future = _add_date_regressors(future)
    fc_all = model.predict(future).set_index("ds")

    # Si LOG_TRANSFORM fuera true, aquí habría que invertir; pero en la config está en false
    feats      = _make_features(fc_all)
    feats_hist = feats.loc[df_ts.index]          # sólo historia
    feats_tail = feats_hist.iloc[-HORIZON_DAYS:] # cola in-sample

    X_ens_tail   = feats_tail.drop(columns=["yhat"])
    resid_pred   = xgb.predict(X_ens_tail)
    yhat_base    = feats_tail["yhat"].values
    y_pred_ens   = yhat_base + resid_pred

    # 7) Gráfica
    plt.figure(figsize=(12,5))
    plt.plot(dates, y_true,      marker="o", label="Real")
    plt.plot(dates, y_pred_lstm, marker="o", label="Quantile LSTM")
    plt.plot(dates, y_pred_ens,  marker="o", label="Ensemble (Prophet+XGB)")
    plt.title("In-Sample: Ventas Reales vs LSTM vs Ensemble")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()