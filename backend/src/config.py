# src/config.py

import os
from dotenv import load_dotenv

# Carga variables de .env
load_dotenv()

# — Rutas de datos ——————————————————————————————
DATA_PATH = os.getenv("DATA_PATH", "./src/data/laloexport.csv")

# — Horario de series ———————————————————————————
# Frecuencia para resample y forecast ("D","W","M",...)
FREQ = os.getenv("FREQ", "D")
# Horizonte de prueba y forecast (días)
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "90"))

# — Parámetros de Prophet ————————————————————————
PROPHET_SEASONALITY_MODE    = os.getenv("PROPHET_SEASONALITY_MODE", "additive")
PROPHET_DAILY_SEASONALITY   = os.getenv("PROPHET_DAILY_SEASONALITY", "true").lower() == "true"
PROPHET_WEEKLY_SEASONALITY  = os.getenv("PROPHET_WEEKLY_SEASONALITY", "true").lower() == "true"
PROPHET_YEARLY_SEASONALITY  = os.getenv("PROPHET_YEARLY_SEASONALITY", "true").lower() == "true"
CP_PRIOR_SCALES             = [float(x) for x in os.getenv("CP_PRIOR_SCALES", "0.001,0.01,0.1,0.2").split(",")]
SEASONALITY_PRIOR_SCALES    = [float(x) for x in os.getenv("SEASONALITY_PRIOR_SCALES", "0.01,0.1,0.5,1.0").split(",")]
CHANGEPNT_RANGE             = float(os.getenv("CHANGEPNT_RANGE", "0.8"))
LOG_TRANSFORM               = os.getenv("LOG_TRANSFORM", "false").lower() == "true"
