import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Tarifas de Acceso RV - nuam",
    layout="wide",
)

st.title("📊 Simulador de Tarifas de Acceso RV - nuam")
st.caption(
    "App para simular tarifas de acceso (fija + variable en bps) para Chile, Colombia y Perú, "
    "y comparar el impacto vs. la situación actual."
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_csv_safe(path: str) -> pd.DataFrame:
    """Carga un CSV manejando codificaciones típicas."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # Si todo falla, lanzamos el error del último intento
    return pd.read_csv(path)


def find_column(df: pd.DataFrame, candidates) -> str:
    """
    Busca una columna en df probando múltiples nombres candidatos
    (ignorando espacios y mayúsculas/minúsculas).
    """
    normalized = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"No se encontró ninguna de las columnas candidatas: {candidates}")


@st.cache_data(show_spinner=False)
def load_data():
    """
    Carga los tres CSV base del modelo.
    Ajusta las rutas si los archivos viven en otra carpeta.
    """
    df_base = load_csv_safe("A.3 BBDD Neg.csv")
    df_params = load_csv_safe("A.6. Clientes MD (segmentado).csv")
    df_resumen = load_csv_safe("3. Negociación.csv")
    return df_base, df_params, df_resumen


def calcular_escenario_acceso(df_base: pd.DataFrame, params_por_pais: dict):
    """
    Recalcula el ingreso de acceso proyectado usando:
        Ingreso_Proyectado = (Monto_USD * bps / 10000) + fijo

    params_por_pais = {
        "Chile":   {"fijo": x, "bps": y},
        "Colombia": {...},
        "Perú": {...},
    }
    """
    df = df_base.copy()

    # Identificación de columnas clave
    col_pais = find_column(df, ["Pais", "País"])
    col_broker = find_column(df, ["Corredor", "Broker", "Corredor / Cliente"])
    col_monto = find_column(df, ["Monto USD", "Monto_USD", "Volumen USD"])
    col_acceso_actual = find_column(
        df,
        ["Cobro Acceso", "Acceso actual", "Acceso_Actual", "Acceso Actual"],
    )

    # Limpieza numérica
    df[col_monto] = pd.to_numeric(df[col_monto], errors="coerce").fillna(0.0)
    df[col_acceso_actual] = pd.to_numeric(df[col_acceso_actual], errors="coerce").fillna(0.0)

    df.rename(
        columns={
            col_pais: "Pais",
            col_broker: "Broker",
            col_monto: "Monto_USD",
            col_acceso_actual: "Acceso_Actual",
        },
        inplace=True,
    )

    # Inicialmente, proyectado = actual
    df["Acceso_Proyectado"] = df["Acceso_Actual"]

    # Aplicamos la fórmula por país
    for pais, cfg in params_por_pais.items():
        fijo = cfg.get("fijo", 0.0) or 0.0
        bps = cfg.get("bps", 0.0) or 0.0

        mask = df["Pais"] == pais
        if not mask.any():
            continue

        monto = df.loc[mask, "Monto_USD"]
        df.loc[mask, "Acceso_Proyectado"] = monto * (bps / 10000.0) + fijo

    # Agregados por país
    agg_pais = (
        df.groupby("Pais")[["Monto_USD", "Acceso_Actual", "Acceso_Proyectado"]]
        .sum()
        .reset_index()
    )

    # BPS resultante
    agg_pais["BPS_Actual"] = np.where(
        agg_pais["Monto_USD"] > 0,
        agg_pais["Acceso_Actual"] / agg_pais["Monto_USD"] * 10000,
        0.0,
    )
    agg_pais["BPS_Proyectado"] = np.where(
        agg_pais["Monto_USD"] > 0,
        agg_pais["Acceso_Proyectado"] / agg_pais["Monto_USD"] * 10000,
        0.0,
    )

    agg_pais["Var_Ingreso_Abs"] = agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]
    agg_pais["Var_Ingreso_%"] = np.where(
        agg_pais["Acceso_Actual"] != 0,
        agg_pais["Var_Ingreso_Abs"] / agg_pais["Acceso_Actual"] * 100,
        0.0,
    )

    # Agregado por broker (para la tabla estilo "3. Negociación")
    agg_broker = (
        df.groupby(["Pais", "Broker"])[["Monto_USD", "Acceso_Actual", "Acceso_Proyectado"]]
        .sum()
        .reset_index()
    )
    agg_broker["Var_Ingreso_%"] = np.where(
        agg_broker["Acceso_Actual"] != 0,
        (agg_broker["Acceso_Proyectado"] - agg_broker["Acceso_Actual"])
        / agg_broker["Acceso_Actual"]
        * 100,
        0.0,
    )

    return df, agg_pais, agg_broker


# ---------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------

with st.spinner("Cargando datos base..."):
    try:
        df_base, df_params, df_resumen = load_data()
    except Exception as e:
        st.error(
            f"❌ No se pudieron cargar los CSV. "
            f"Verifica que existan en la carpeta de la app.\n\nError: {e}"
        )
        st.stop()

st.success("Datos cargados correctamente.")

# ---------------------------------------------------------
# Sidebar: Parámetros por país
# ---------------------------------------------------------

st.sidebar.header("⚙️ Parámetros de Acceso")

st.sidebar.markdown("### Chile")
fixed_cl = st.sidebar.number_input(
    "Tarifa fija Chile (USD)",
    min_value=0.0,
    value=500.0,
    step=50.0,
)
bps_cl = st.sidebar.number_input(
    "Tarifa variable Chile (bps)",
    min_value=0.0,
    value=0.0,
    step=5.0,
)

st.sidebar.markdown("### Colombia")
fixed_co = st.sidebar.number_input(
    "Tarifa fija Colombia (USD)",
    min_value=0.0,
    value=500.0,
    step=50.0,
)
bps_co = st.sidebar.number_input(
    "Tarifa variable Colombia (bps)",
    min_value=0.0,
    value=0.0,
    step=5.0,
)

st.sidebar.markdown("### Perú")
fixed_pe = st.sidebar.number_input(
    "Tarifa fija Perú (USD)",
    min_value=0.0,
    value=500.0,
    step=50.0,
)
bps_pe = st.sidebar.number_input(
    "Tarifa variable Perú (bps)",
    min_value=0.0,
    value=0.0,
    step=5.0,
)

st.sidebar.markdown("---")
recalc = st.sidebar.button("🔄 Recalcular impacto")

# Nota: en Streamlit el script se re-ejecuta en cada cambio,
# así que el botón es más bien "cosmético". Aun así lo usamos
# para mostrar un mensajito.
if recalc:
    st.sidebar.success("Impacto recalculado con los parámetros actuales.")

# ---------------------------------------------------------
# Cálculo del escenario
# ---------------------------------------------------------

params = {
    "Chile": {"fijo": fixed_cl, "bps": bps_cl},
    "Colombia": {"fijo": fixed_co, "bps": bps_co},
    "Perú": {"fijo": fixed_pe, "bps": bps_pe},
}

df_detalle, df_pais, df_broker = calcular_escenario_acceso(df_base, params)

# ---------------------------------------------------------
# KPIs generales por país
# ---------------------------------------------------------

st.subheader("🏁 KPIs de Acceso por País")

# Aseguramos que estén los tres países como filas, aunque tengan 0
for p in ["Chile", "Colombia", "Perú"]:
    if p not in df_pais["Pais"].values:
        df_pais = pd.concat(
            [
                df_pais,
                pd.DataFrame(
                    {
                        "Pais": [p],
                        "Monto_USD": [0.0],
                        "Acceso_Actual": [0.0],
                        "Acceso_Proyectado": [0.0],
                        "BPS_Actual": [0.0],
                        "BPS_Proyectado": [0.0],
                        "Var_Ingreso_Abs": [0.0],
                        "Var_Ingreso_%": [0.0],
                    }
                ),
            ],
            ignore_index=True,
        )

df_pais = df_pais.set_index("Pais").loc[["Chile", "Colombia", "Perú"]].reset_index()

col1, col2, col3 = st.columns(3)

for col, pais in zip([col1, col2, col3], ["Chile", "Colombia", "Perú"]):
    row = df_pais[df_pais["Pais"] == pais].iloc[0]
    var_pct = row["Var_Ingreso_%"]
    bps_proj = row["BPS_Proyectado"]

    col.metric(
        label=f"{pais} – Variación Ingresos Acceso (%)",
        value=f"{var_pct:,.2f} %",
    )
    col.metric(
        label=f"{pais} – BPS Acceso proyectado",
        value=f"{bps_proj:,.1f} bps",
    )

st.markdown("---")

# ---------------------------------------------------------
# Tabla detallada estilo "3. Negociación"
# ---------------------------------------------------------

st.subheader("📋 Detalle por Broker")

tabs = st.tabs(["Chile", "Colombia", "Perú", "Todos"])

def tabla_brokers_por_pais(df_broker, pais=None):
    df = df_broker.copy()
    if pais is not None:
        df = df[df["Pais"] == pais]

    df_view = df[["Broker", "Monto_USD", "Acceso_Actual", "Acceso_Proyectado", "Var_Ingreso_%"]].copy()
    df_view.rename(
        columns={
            "Broker": "Broker / Corredor",
            "Monto_USD": "Monto negociado (USD)",
            "Acceso_Actual": "Ingreso real (Acceso actual)",
            "Acceso_Proyectado": "Ingreso proyectado (Simulado)",
            "Var_Ingreso_%": "Var % Ingreso",
        },
        inplace=True,
    )
    return df_view.sort_values("Monto negociado (USD)", ascending=False)


with tabs[0]:
    st.markdown("**Chile** – Detalle por broker")
    st.dataframe(tabla_brokers_por_pais(df_broker, "Chile"), use_container_width=True)

with tabs[1]:
    st.markdown("**Colombia** – Detalle por broker")
    st.dataframe(tabla_brokers_por_pais(df_broker, "Colombia"), use_container_width=True)

with tabs[2]:
    st.markdown("**Perú** – Detalle por broker")
    st.dataframe(tabla_brokers_por_pais(df_broker, "Perú"), use_container_width=True)

with tabs[3]:
    st.markdown("**Todos los países** – Detalle consolidado por broker")
    st.dataframe(tabla_brokers_por_pais(df_broker, None), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Gráfico de barras: Actual vs Proyectado por país
# ---------------------------------------------------------

st.subheader("📈 Ingreso de Acceso: Actual vs Proyectado por País")

df_chart = df_pais[["Pais", "Acceso_Actual", "Acceso_Proyectado"]].set_index("Pais")
st.bar_chart(df_chart)

st.markdown("---")

st.caption(
    "Modelo simplificado: Ingreso_Proyectado = Monto_USD * (bps / 10,000) + Tarifa fija por fila. "
    "La estructura está preparada para extenderla luego a tramos (T, U, W) usando df_params."
)
