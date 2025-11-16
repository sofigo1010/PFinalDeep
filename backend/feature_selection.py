import os
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data.ingestion       import load_and_prepare_data
from src.data.preprocessing   import preprocess_sales_data
from src.config               import FREQ

def smape(y_true, y_pred):
    mask = y_true > 1e-3
    return 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) /
                        (np.abs(y_true[mask]) + np.abs(y_pred[mask])))

def create_features(df, window, alpha=None):
    df = df.copy()
    lower, upper = df["sales"].quantile([0.01, 0.99])
    df["sales_smooth"] = df["sales"].clip(lower, upper)
    df["sales_log"]    = np.log1p(df["sales_smooth"])
    if alpha is None:
        df[f"rm_{window}"]   = df["sales_log"].rolling(window, min_periods=1).mean()
        df[f"rstd_{window}"] = df["sales_log"].rolling(window, min_periods=1).std().fillna(0)
    else:
        df[f"ewm_{window}"]  = df["sales_log"].ewm(alpha=alpha, adjust=False).mean()
    return df

def evaluate_window(df_ts, window, use_ewm=False, alpha=None):
    df_feat   = create_features(df_ts, window, alpha if use_ewm else None)
    feat_cols = [c for c in df_feat.columns if c.startswith(("rm_","rstd_","ewm_"))]
    X         = df_feat[feat_cols].values
    y_log     = df_feat["sales_log"].values
    tscv      = TimeSeriesSplit(n_splits=5)
    scores    = []
    for tr, va in tscv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y_log[tr], y_log[va]
        m = RandomForestRegressor(n_estimators=50, random_state=42)
        m.fit(Xtr, ytr)
        p = m.predict(Xva)
        scores.append(smape(np.expm1(yva), np.expm1(p)))
    return np.mean(scores)

def grid_search_windows(df_ts):
    windows = [3, 7, 14, 21, 30, 60]
    res     = {}
    for w in windows:
        res[f"rm_std_{w}"] = evaluate_window(df_ts, w, use_ewm=False)
    for w in windows:
        a = 2/(w+1)
        res[f"ewm_alpha_{a:.2f}"] = evaluate_window(df_ts, w, use_ewm=True, alpha=a)
    best = min(res, key=res.get)
    return best, res

def bayesian_search(df_ts, n_trials=30):
    def obj(trial):
        w = trial.suggest_int("window", 3, 60)
        a = trial.suggest_float("alpha", 0.01, 0.5)
        return evaluate_window(df_ts, w, use_ewm=True, alpha=a)
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials)
    return study.best_params, study.best_value

def remove_redundant_features(df, feature_cols, corr_thresh=0.9, vif_thresh=5.0):
    X = df[feature_cols].copy()
    dropped_corr = []
    if X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dropped_corr = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        X = X.drop(columns=dropped_corr)

    dropped_vif = []
    if X.shape[1] > 1:
        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        dropped_vif = X.columns[np.array(vif_vals) > vif_thresh].tolist()
        X = X.drop(columns=dropped_vif)

    dropped_lasso = []
    if X.shape[1] > 1:
        lasso = LassoCV(cv=5).fit(X, df["sales_log"])
        dropped_lasso = X.columns[np.isclose(lasso.coef_, 0)].tolist()
        X = X.drop(columns=dropped_lasso)

    # PCA (si queda al menos 1 feature)
    if X.shape[1] >= 1:
        pca    = PCA(n_components=0.95)
        X_pca  = pca.fit_transform(X)
        n_comp = pca.n_components_
    else:
        X_pca  = np.empty((X.shape[0], 0))
        n_comp = 0

    return X, X_pca, {
        "dropped_corr":   dropped_corr,
        "dropped_vif":    dropped_vif,
        "dropped_lasso":  dropped_lasso,
        "pca_components": n_comp
    }

if __name__ == "__main__":
    df_raw = load_and_prepare_data()
    df_ts  = preprocess_sales_data(df_raw, freq=FREQ)

    # 1) Grid Search
    best_w, grid_res = grid_search_windows(df_ts)
    print("Grid Search best:", best_w)
    for k, v in grid_res.items():
        print(f"  {k}: {v:.2f}% sMAPE")

    # 2) Bayesian Search
    best_params, best_sc = bayesian_search(df_ts, n_trials=20)
    print("Optuna best params:", best_params, f"{best_sc:.2f}% sMAPE")

    # 3) Feature Reduction
    df_feat   = create_features(df_ts, best_params["window"], alpha=best_params["alpha"])
    fc        = [c for c in df_feat.columns if c.startswith(("rm_","rstd_","ewm_"))]
    X_red, X_pca, info = remove_redundant_features(df_feat, fc)
    print("Feature reduction info:", info)
