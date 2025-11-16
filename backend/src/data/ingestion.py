import pandas as pd
from typing import IO, Union
from src.config import DATA_PATH

DATE_CANDIDATES = ["Order Date", "Paid at", "Created at", "Processed at"]
SALES_CANDIDATES = ["Sales", "Total", "Subtotal"]


def _pick_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    """
    Devuelve el primer nombre de columna que exista en el DataFrame
    dentro de la lista candidates. Si no encuentra, lanza error claro.
    kind se usa solo para mensaje de error ("fecha", "ventas", etc.).
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró ninguna columna de {kind}. "
        f"Se esperaba alguna de: {candidates}. "
        f"Columnas encontradas en el CSV: {list(df.columns)}"
    )


def load_sales_data(source: Union[str, IO] = None) -> pd.DataFrame:
    """
    Lee el CSV (Superstore o export de órdenes tipo Shopify),
    detecta columnas de fecha y venta, las renombra a date/sales,
    agrupa ventas diarias y, si existe Profit, también agrupa profit.
    """
    path = source or DATA_PATH
    df = pd.read_csv(path)

    # Detectar columnas de fecha y ventas según el tipo de archivo
    date_col = _pick_column(df, DATE_CANDIDATES, kind="fecha")
    sales_col = _pick_column(df, SALES_CANDIDATES, kind="ventas")

    # Renombrar a nombres estándar internos
    rename_map = {
        date_col: "date",
        sales_col: "sales",
    }
    if "Profit" in df.columns:
        rename_map["Profit"] = "profit"

    df = df.rename(columns=rename_map)

    # Filtrar solo órdenes pagadas si existe Financial Status (caso Shopify)
    if "Financial Status" in df.columns:
        fs = df["Financial Status"].astype(str).str.lower()
        df = df[fs.isin(["paid", "partially_paid", "partially_refunded"])]

    # Fecha → datetime con UTC para unificar zonas, luego quitar tz y normalizar a día
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_convert(None)   # quita la zona horaria
    df["date"] = df["date"].dt.normalize()        # deja solo la fecha (00:00:00)

    # Asegurar que sales es numérico
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    # Limpiar filas sin fecha o sin ventas
    df = df.dropna(subset=["date", "sales"])

    # Si existe profit, limpiar NaN ahí también
    if "profit" in df.columns:
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df = df.dropna(subset=["profit"])

    # Agrupar por día (sumar ventas, y profit si existe)
    agg_dict = {"sales": "sum"}
    if "profit" in df.columns:
        agg_dict["profit"] = "sum"

    df_daily = df.groupby("date", as_index=False).agg(agg_dict)
    return df_daily


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    return df


def load_and_prepare_data(source: Union[str, IO] = None) -> pd.DataFrame:
    df = load_sales_data(source)
    return clean_sales_data(df)