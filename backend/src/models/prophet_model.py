# src/models/prophet_model.py

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from joblib import dump, load
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from src.config import (
    PROPHET_SEASONALITY_MODE,
    PROPHET_DAILY_SEASONALITY,
    PROPHET_WEEKLY_SEASONALITY,
    PROPHET_YEARLY_SEASONALITY,
    CP_PRIOR_SCALES,
    SEASONALITY_PRIOR_SCALES,
    LOG_TRANSFORM,
    CHANGEPNT_RANGE,
)
from src.data.preprocessing import inverse_log_transform

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _add_date_regressors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"]  = df["ds"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["month"]      = df["ds"].dt.month
    df["quarter"]    = df["ds"].dt.quarter
    return df


def _get_special_events(years):
    events = []
    for y in years:
        # Black Friday: cuarto viernes de noviembre, con efecto -1 a +2 días
        nov     = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
        fridays = nov[nov.weekday == 4]
        bf      = fridays[3]
        cm      = bf + pd.Timedelta(days=3)  # Cyber Monday
        events.append({
            "holiday": "black_friday",
            "ds": bf,
            "lower_window": -1, "upper_window": 2
        })
        events.append({
            "holiday": "cyber_monday",
            "ds": cm,
            "lower_window": 0, "upper_window": 1
        })
    return pd.DataFrame(events)


def train_prophet(
    df: pd.DataFrame,
    model_path: Path = MODEL_DIR / "prophet_model.joblib",
    changepoint_prior_scale: float = None,
    seasonality_prior_scale: float = None,
    **prophet_kwargs
) -> Prophet:
    # 1) Preparo ds / y
    df_prop = df.reset_index().rename(columns={"date": "ds", "sales": "y"})
    if LOG_TRANSFORM:
        df_prop["y"] = np.log(df_prop["y"] + 1)

    # 2) Regresores temporales
    df_prop = _add_date_regressors(df_prop)

    # 3) Hiperparámetros CPS / SPS
    cps = changepoint_prior_scale or CP_PRIOR_SCALES[0]
    sps = seasonality_prior_scale or SEASONALITY_PRIOR_SCALES[0]

    # 4) Inicializo Prophet (lineal) con seasonality multiplicativa
    m = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        daily_seasonality=PROPHET_DAILY_SEASONALITY,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        changepoint_range=0.95,
        n_changepoints=25,
        mcmc_samples=0,
        **prophet_kwargs,
    )

    # 5) Regresores extra
    for reg in ["dayofweek","is_weekend","month","quarter"]:
        m.add_regressor(reg)

    # 6) Estacionalidades finas
    m.add_seasonality(
        name="daily",
        period=1,
        fourier_order=3,
        prior_scale=0.02
    )
    m.add_seasonality(
        name="weekly",
        period=7,
        fourier_order=10,
        prior_scale=0.2
    )
    m.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=8,
        prior_scale=0.1
    )
    m.add_seasonality(
        name="quarterly",
        period=91.31,
        fourier_order=5,
        prior_scale=0.05
    )

    # 7) Festivos EE.UU. + eventos especiales
    m.add_country_holidays(country_name="US")
    years   = sorted(df_prop["ds"].dt.year.unique())
    special = _get_special_events(years)
    m.holidays = pd.concat([m.holidays, special], ignore_index=True)

    # 8) Entrenar y guardar
    m.fit(df_prop)
    dump(m, model_path)
    return m


def predict_prophet(
    model: Prophet = None,
    periods: int = None,
    freq: str = None,
    model_path: Path = MODEL_DIR / "prophet_model.joblib",
) -> pd.DataFrame:
    from src.config import HORIZON_DAYS, FREQ

    if model is None:
        model = load(model_path)

    periods = periods or HORIZON_DAYS
    freq    = freq or FREQ

    # 1) Conjunto futuro + regresores
    future = model.make_future_dataframe(periods=periods, freq=freq)
    future = _add_date_regressors(future)

    # 2) Predecir
    fcst = model.predict(future)

    # 3) Invertir log si aplica
    if LOG_TRANSFORM:
        for c in ["yhat","yhat_lower","yhat_upper"]:
            fcst[c] = inverse_log_transform(fcst[c])

    return fcst[["ds","yhat","yhat_lower","yhat_upper"]]


def load_model(
    model_path: Path = MODEL_DIR / "prophet_model.joblib"
) -> Prophet:
    return load(model_path)


def tune_prophet(
    df: pd.DataFrame,
    initial: str = "730 days",
    period:  str = "90 days",
    horizon: str = "90 days"
) -> dict:
    """
    Grid search optimizando MAPE con validación cruzada.
    Se adapta automáticamente a la longitud de la serie:
    - Si hay muy pocos datos, devuelve hiperparámetros por defecto sin CV.
    - Si se usan los defaults de initial/period/horizon, se recalculan
      en función del número de días disponibles.
    """
    df_prop = df.reset_index().rename(columns={"date":"ds","sales":"y"})
    if LOG_TRANSFORM:
        df_prop["y"] = np.log(df_prop["y"] + 1)
    df_prop = _add_date_regressors(df_prop)

    # ---- calcular cuántos días de historia hay ----
    n_days = (df_prop["ds"].max() - df_prop["ds"].min()).days
    # Si hay MUY pocos datos, saltarse el CV y usar defaults
    if n_days < 120:
        return {
            "mape": float("nan"),
            "cps": CP_PRIOR_SCALES[0],
            "sps": SEASONALITY_PRIOR_SCALES[0],
        }

    # Si el usuario dejó los defaults, ajustar ventanas automáticamente
    if initial == "730 days" and horizon == "90 days" and period == "90 days":
        # horizonte ≈ 1/4 de la serie, entre 7 y 30 días
        horizon_days  = max(7, min(30, n_days // 4))
        # initial: resto de la historia, pero al menos 30 días
        initial_days  = max(30, n_days - 2 * horizon_days)
        # period: igual que el horizonte (separación entre cortes)
        period_days   = max(7, horizon_days)

        initial = f"{initial_days} days"
        horizon = f"{horizon_days} days"
        period  = f"{period_days} days"

    best = {"mape": float("inf"), "cps": None, "sps": None}

    for cps, sps in product(CP_PRIOR_SCALES, SEASONALITY_PRIOR_SCALES):
        m = Prophet(
            growth="linear",
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            changepoint_range=0.95,
            n_changepoints=25,
            mcmc_samples=0,
        )
        for reg in ["dayofweek","is_weekend","month","quarter"]:
            m.add_regressor(reg)
        m.add_seasonality("weekly",    period=7,    fourier_order=3)
        m.add_seasonality("monthly",   period=30.5, fourier_order=5)
        m.add_seasonality("quarterly", period=91.31, fourier_order=3)

        m.fit(df_prop)
        df_cv = cross_validation(
            m, initial=initial, period=period, horizon=horizon, parallel="processes"
        )
        perf = performance_metrics(df_cv)
        mape = perf["mape"].mean()
        if mape < best["mape"]:
            best.update({"mape": mape, "cps": cps, "sps": sps})

    return best
