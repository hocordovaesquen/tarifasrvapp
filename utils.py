from typing import Iterable, Optional, Set

import pandas as pd


def find_header_row(df: pd.DataFrame, required_cols: Set[str], search_depth: int = 20) -> Optional[int]:
    """Encuentra el índice de fila que contiene todos los encabezados requeridos.

    Busca en las primeras `search_depth` filas para evitar recorrer archivos enormes.
    """
    required_lower = {c.lower() for c in required_cols}
    for idx in range(min(search_depth, len(df))):
        row_values = {str(c).strip().lower() for c in df.iloc[idx].tolist()}
        if required_lower.issubset(row_values):
            return idx
    return None


def format_currency(value: float) -> str:
    return f"${value:,.2f} USD"


def format_bps(value: float) -> str:
    return f"{value:.2f} bps"


def validate_columns(df: pd.DataFrame, expected: Iterable[str]) -> Optional[str]:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return f"Faltan columnas: {', '.join(missing)}"
    return None