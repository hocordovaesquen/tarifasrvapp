from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class SimulationConfig:
    fx_rates: Dict[str, float]
    tranche_limits: List[float]
    tranche_bps: List[float]


# ---------------------------------------------------------------------------
# Conversión FX y tramos
# ---------------------------------------------------------------------------
def convert_to_usd(amount: float, currency: str, fx_rates: Dict[str, float]) -> float:
    """Convierte montos a USD usando las tasas provistas.

    Asume que las monedas LATAM están expresadas en moneda local por USD,
    por lo que la conversión es amount / tasa.
    """
    clean_currency = str(currency).strip().upper()
    rate = fx_rates.get(clean_currency, 1.0)
    if rate == 0:
        return 0.0
    return amount / rate


def apply_tranches(volume: float, tranche_limits: Iterable[float], tranche_bps: Iterable[float]) -> float:
    """Calcula el BPS correspondiente según volumen y tramos definidos."""
    limits = list(tranche_limits)
    bps = list(tranche_bps)
    if len(bps) != len(limits) + 1:
        raise ValueError("La cantidad de tarifas debe ser igual a límites + 1")

    if volume <= limits[0]:
        return bps[0]
    if len(limits) > 1 and volume <= limits[1]:
        return bps[1]
    return bps[-1]


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
def calculate_simulation(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Calcula volumen USD, tarifa aplicada e ingreso estimado para cada fila."""
    required_cols = ["Monto Local", "Moneda", "Corredor"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en la hoja de negociación: {', '.join(missing)}")

    work_df = df.copy()
    work_df["Volumen_USD_Simulado"] = work_df.apply(
        lambda row: convert_to_usd(row["Monto Local"], row["Moneda"], config.fx_rates), axis=1
    )

    work_df["Tarifa_BPS_Simulada"] = work_df["Volumen_USD_Simulado"].apply(
        lambda vol: apply_tranches(vol, config.tranche_limits, config.tranche_bps)
    )

    work_df["Ingreso_USD_Simulado"] = work_df["Volumen_USD_Simulado"] * (
        work_df["Tarifa_BPS_Simulada"] / 10000
    )

    return work_df


def summarize_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa resultados simulados por país."""
    if "País" not in df.columns:
        return pd.DataFrame()
    cols = ["Volumen_USD_Simulado", "Ingreso_USD_Simulado"]
    available = [c for c in cols if c in df.columns]
    return df.groupby("País")[available].sum()


def summarize_by_broker(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa resultados simulados por corredor."""
    cols = ["Volumen_USD_Simulado", "Ingreso_USD_Simulado"]
    available = [c for c in cols if c in df.columns]
    return df.groupby("Corredor")[available].sum()