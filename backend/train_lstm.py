#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
import holidays
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.data.ingestion     import load_and_prepare_data
from src.data.preprocessing import preprocess_sales_data
from src.models.LSTM_model  import LSTMModel
from src.config             import FREQ

def enrich_features(df):
    """
    Agrega variables exógenas, calendario avanzado,
    codificación cíclica y estadísticas rodantes.
    """
    # promociones
    promo_path = "data/promotions.csv"
    if os.path.exists(promo_path):
        promo = pd.read_csv(promo_path, parse_dates=["date"])
        promo = promo.set_index("date").reindex(df.index, fill_value=0)
        df["promo_flag"] = promo["promo_flag"]
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

    # suavizado y log-transform
    lower, upper = df["sales"].quantile([0.01, 0.99])
    df["sales_smooth"] = df["sales"].clip(lower, upper)
    df["sales_log"]    = np.log1p(df["sales_smooth"])

    # estadísticas rodantes de sales_log
    for w in [7, 14, 30]:
        df[f"rm_{w}"]   = df["sales_log"].rolling(window=w, min_periods=1).mean()
        df[f"rstd_{w}"] = df["sales_log"].rolling(window=w, min_periods=1).std().fillna(0)

    # codificación cíclica
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

def quantile_loss(pred, target, q=0.5):
    err = target - pred
    return torch.max(q * err, (q - 1) * err).mean()

def main():
    # 1) Datos & preprocesado
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)
    df     = enrich_features(df_ts.copy())

    # 2) Matriz de features
    features = [
        "sales_log", "promo_flag", "is_holiday",
        "is_month_start", "is_month_end",
        "is_quarter_start", "is_quarter_end",
        "days_to_holiday",
        "rm_7", "rstd_7",
        "rm_14", "rstd_14",
        "rm_30", "rstd_30",
        "month_sin", "month_cos",
        "dow_sin", "dow_cos"
    ]
    X_df       = df[features]
    input_size = X_df.shape[1]

    # 3) Escalado
    scaler     = MinMaxScaler()
    scaled_all = scaler.fit_transform(X_df.values)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/lstm_scaler.pkl")

    # 4) Secuencias
    SEQ_LEN    = 60
    target_idx = features.index("sales_log")
    X, y_log   = create_sequences(scaled_all, SEQ_LEN, target_idx)

    # 5) Train/Val split
    split      = int(0.8 * len(X))
    X_train, y_train = X[:split], y_log[:split]
    X_val,   y_val   = X[split:], y_log[split:]

    # 6) DataLoaders
    train_ds     = TensorDataset(torch.from_numpy(X_train).float(),
                                 torch.from_numpy(y_train).float())
    val_ds       = TensorDataset(torch.from_numpy(X_val).float(),
                                 torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    # 7) Entrenamiento
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = LSTMModel(input_size=input_size, hidden_size=128,
                          num_layers=2, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val, patience, counter = float("inf"), 10, 0
    for epoch in range(1, 101):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss  = quantile_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds  = model(xb).squeeze()
                val_losses.append(quantile_loss(preds, yb).item())
        mean_val = np.mean(val_losses)
        scheduler.step(mean_val)

        print(f"Epoch {epoch:03d} | train_loss={np.mean(train_losses):.6f} | val_loss={mean_val:.6f}")
        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), "models/lstm_quantile.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    print("✅ Entrenamiento completado — modelo guardado en models/lstm_quantile.pth")

if __name__ == "__main__":
    main()
