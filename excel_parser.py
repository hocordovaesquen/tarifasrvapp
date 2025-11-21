from typing import Optional, Tuple

import pandas as pd

from utils import find_header_row


CUSTOMER_JOURNEY_SHEET = "6. Customer Journey"
END_TO_END_SHEET = "EndToEnd"
NEGOTIATION_SHEET = "A.3 BBDD Neg"


# ---------------------------------------------------------------------------
# Customer Journey
# ---------------------------------------------------------------------------
def parse_customer_journey(xls: pd.ExcelFile) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if CUSTOMER_JOURNEY_SHEET not in xls.sheet_names:
        return None, f"No se encontró la hoja '{CUSTOMER_JOURNEY_SHEET}'."

    raw_df = pd.read_excel(xls, sheet_name=CUSTOMER_JOURNEY_SHEET, header=None)
    header_index = find_header_row(raw_df, {"UN", "Producto"})
    if header_index is None:
        return None, "No se pudo localizar la tabla 'INGRESOS POR PRODUCTO' en Customer Journey."

    df = pd.read_excel(xls, sheet_name=CUSTOMER_JOURNEY_SHEET, header=header_index)
    expected_cols = {"UN", "Producto", "Ingreso Actual", "Ingreso Propuesta"}
    missing = expected_cols - set(df.columns)
    if missing:
        return None, f"Faltan columnas en Customer Journey: {', '.join(missing)}"

    df = df[list(expected_cols)].dropna(subset=["UN", "Producto"])
    df["Ingreso Actual"] = pd.to_numeric(df["Ingreso Actual"], errors="coerce").fillna(0)
    df["Ingreso Propuesta"] = pd.to_numeric(df["Ingreso Propuesta"], errors="coerce").fillna(0)
    df["UN"] = df["UN"].astype(str)
    df["Producto"] = df["Producto"].astype(str)
    return df, None


# ---------------------------------------------------------------------------
# EndToEnd
# ---------------------------------------------------------------------------
def parse_end_to_end(xls: pd.ExcelFile) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if END_TO_END_SHEET not in xls.sheet_names:
        return None, f"No se encontró la hoja '{END_TO_END_SHEET}'."

    raw_df = pd.read_excel(xls, sheet_name=END_TO_END_SHEET, header=None)
    header_index = find_header_row(raw_df, {"Bloque", "Subbloque"})
    if header_index is None:
        return None, "No se pudo localizar la tabla de resumen EndToEnd."

    df = pd.read_excel(xls, sheet_name=END_TO_END_SHEET, header=header_index)

    expected_cols = {
        "Bloque",
        "Subbloque",
        "Colombia_tarifa",
        "Perú_tarifa",
        "Chile_tarifa",
        "Colombia_ingresos",
        "Perú_ingresos",
        "Chile_ingresos",
    }
    # Intento de normalizar columnas que suelen venir con sufijos
    df.columns = [str(c).strip() for c in df.columns]
    column_map = {}
    for col in df.columns:
        lower = col.lower()
        if "colombia" in lower and "bps" in lower:
            column_map[col] = "Colombia_tarifa"
        elif "per" in lower and "bps" in lower:
            column_map[col] = "Perú_tarifa"
        elif "chile" in lower and "bps" in lower:
            column_map[col] = "Chile_tarifa"
        elif "colombia" in lower and "ingreso" in lower:
            column_map[col] = "Colombia_ingresos"
        elif "per" in lower and "ingreso" in lower:
            column_map[col] = "Perú_ingresos"
        elif "chile" in lower and "ingreso" in lower:
            column_map[col] = "Chile_ingresos"
    df = df.rename(columns=column_map)

    missing = expected_cols - set(df.columns)
    if missing:
        return None, f"Faltan columnas en EndToEnd: {', '.join(missing)}"

    numeric_cols = [c for c in df.columns if "ingresos" in c or "tarifa" in c]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[[
        "Bloque",
        "Subbloque",
        "Colombia_tarifa",
        "Perú_tarifa",
        "Chile_tarifa",
        "Colombia_ingresos",
        "Perú_ingresos",
        "Chile_ingresos",
    ]].dropna(subset=["Bloque", "Subbloque"])

    df["Bloque"] = df["Bloque"].astype(str)
    df["Subbloque"] = df["Subbloque"].astype(str)
    return df, None


# ---------------------------------------------------------------------------
# Negociación (A.3 BBDD Neg)
# ---------------------------------------------------------------------------
def parse_negotiation_base(xls: pd.ExcelFile) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if NEGOTIATION_SHEET not in xls.sheet_names:
        return None, f"No se encontró la hoja '{NEGOTIATION_SHEET}'."

    df = pd.read_excel(xls, sheet_name=NEGOTIATION_SHEET, header=None)
    header_index = find_header_row(df, {"Corredor", "Monto Local", "Moneda"})
    if header_index is None:
        return None, "No se pudo localizar la cabecera en la hoja de negociación."

    df = pd.read_excel(xls, sheet_name=NEGOTIATION_SHEET, header=header_index)

    required_cols = {"País", "Corredor", "Monto Local", "Moneda"}
    missing = required_cols - set(df.columns)
    if missing:
        return None, f"Faltan columnas obligatorias en la hoja de negociación: {', '.join(missing)}"

    df = df[list(required_cols)].dropna(subset=["Corredor", "Monto Local"])
    df["Monto Local"] = pd.to_numeric(df["Monto Local"], errors="coerce").fillna(0)
    df["Corredor"] = df["Corredor"].astype(str)
    return df, None