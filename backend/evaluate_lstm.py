import os
import numpy as np
import pandas as pd
import joblib
import holidays
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from src.data.ingestion       import load_and_prepare_data
from src.data.preprocessing   import preprocess_sales_data
from src.models.LSTM_model    import LSTMModel
from src.config               import FREQ, HORIZON_DAYS

def enrich_features(df):
    # idéntico al entrenamiento
    promo_path = "data/promotions.csv"
    if os.path.exists(promo_path):
        promo = pd.read_csv(promo_path, parse_dates=["date"])
        promo = promo.set_index("date").reindex(df.index, fill_value=0)
        df["promo_flag"] = promo["promo_flag"]
    else:
        df["promo_flag"] = 0

    us_holidays = holidays.US()
    df["is_holiday"] = df.index.to_series().apply(lambda d: 1 if d in us_holidays else 0)
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"]   = df.index.is_month_start.astype(int)
    df["is_month_end"]     = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"]   = df.index.is_quarter_end.astype(int)

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

def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

def filtered_mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    mask = y_true > 1e-3
    return 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) /
                         (np.abs(y_true[mask]) + np.abs(y_pred[mask])))

def main():
    # 1) Carga y preprocesado
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)
    df     = enrich_features(df_ts.copy())

    # 2) Matriz de features
    features = [
        "sales_log", "promo_flag", "is_holiday",
        "is_month_start", "is_month_end",
        "is_quarter_start", "is_quarter_end",
        "days_to_holiday", "rm_7", "rstd_7",
        "rm_14", "rstd_14", "rm_30", "rstd_30",
        "month_sin", "month_cos", "dow_sin", "dow_cos"
    ]
    X_df       = df[features]
    input_size = X_df.shape[1]

    # 3) Escalado (mismo scaler del entrenamiento)
    scaler      = joblib.load("models/lstm_scaler.pkl")
    data_scaled = scaler.transform(X_df.values)

    # 4) Secuencias
    SEQ_LEN    = 60
    target_idx = features.index("sales_log")
    X_all, y_scaled = create_sequences(data_scaled, SEQ_LEN, target_idx)

    # 5) Test set = últimos HORIZON_DAYS
    X_test       = X_all[-HORIZON_DAYS:]
    y_scaled_test = y_scaled[-HORIZON_DAYS:]

    # 6) Carga modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMModel(input_size=input_size, hidden_size=128,
                       num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("models/lstm_quantile.pth", map_location=device))
    model.eval()

    # 7) Predicción (en escala del scaler)
    preds_scaled = []
    loader    = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                      torch.zeros(len(X_test))),
        batch_size=32
    )
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds_scaled.extend(model(xb).cpu().numpy().flatten())
    preds_scaled = np.array(preds_scaled)

    # 8) Back-transform: desescalar -> tomar sales_log -> expm1
    # Predicciones
    tmp_pred = np.zeros((len(preds_scaled), len(features)))
    tmp_pred[:, target_idx] = preds_scaled
    inv_pred = scaler.inverse_transform(tmp_pred)
    sales_log_pred = inv_pred[:, target_idx]
    y_pred = np.expm1(sales_log_pred)

    # Valores reales
    tmp_true = np.zeros_like(tmp_pred)
    tmp_true[:, target_idx] = y_scaled_test
    inv_true = scaler.inverse_transform(tmp_true)
    sales_log_true = inv_true[:, target_idx]
    y_true = np.expm1(sales_log_true)

    # 9) Métricas
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = filtered_mape(y_true, y_pred)
    smp  = smape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n--- Métricas Quantile LSTM (últimos {HORIZON_DAYS} días) ---")
    print(f"MAE:   {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAPE*: {mape:.2f}%")
    print(f"sMAPE: {smp:.2f}%")
    print(f"R²:    {r2:.3f}")

    # 10) Gráfico
    idx = df_ts.index[-HORIZON_DAYS:]
    plt.figure(figsize=(10,4))
    plt.plot(idx, y_true, label="True")
    plt.plot(idx, y_pred, label="Quantile LSTM")
    plt.legend()
    plt.title("Quantile LSTM Forecast vs Actual (in-sample cola)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()