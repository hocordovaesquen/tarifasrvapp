import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional, Dict


# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Tarifas de Acceso RV - nuam",
    layout="wide",
)

st.title("📊 Simulador de Tarifas de Acceso RV - nuam")
st.caption(
    "Simula la tarifa de Acceso RV (por tramos mensuales) para Chile, Colombia y Perú "
    "a partir del modelo Excel. Compara ingreso real vs proyectado por broker y por país."
)


# ---------------------------------------------------------
# Funciones auxiliares de carga
# ---------------------------------------------------------

def load_bbdd_from_bytes(data: bytes, sheet_name: str = "A.3 BBDD Neg") -> pd.DataFrame:
    """
    Carga la hoja A.3 BBDD Neg.
    En tu archivo, el encabezado está en la fila 7 (index 6).
    """
    df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=6)
    return df


def load_param_sheet(excel_bytes: bytes) -> pd.DataFrame:
    """Hoja 1. Parametros sin encabezados."""
    return pd.read_excel(BytesIO(excel_bytes), sheet_name="1. Parametros", header=None)


def load_negociacion_sheet(excel_bytes: bytes) -> pd.DataFrame:
    """
    Hoja 3. Negociación donde están:
        Corredor / Real / Proyectado / Var% de Acceso.
    El header útil empieza en la fila 9 (index 8).
    """
    df = pd.read_excel(BytesIO(excel_bytes), sheet_name="3. Negociación", header=8)
    return df


# ---------------------------------------------------------
# Tramos de acceso desde 1. Parametros
# ---------------------------------------------------------

def get_tramos_acceso(df_params: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Lee los tramos de Acceso OP Directo institucional desde 1. Parametros.

    Bloque de tramos (filas 90-92) y columnas:
        - Colombia: Min = col 19, Max = col 20, Fija_USD = col 22
        - Perú:     Min = col 23, Max = col 24, Fija_USD = col 26
        - Chile:    Min = col 27, Max = col 28, Fija_USD = col 30
    """
    blocks = {
        "Colombia": (19, 20, 22),
        "Perú": (23, 24, 26),
        "Chile": (27, 28, 30),
    }

    tramos_por_pais: Dict[str, pd.DataFrame] = {}

    for pais, (c_min, c_max, c_fija) in blocks.items():
        rows = [90, 91, 92]  # Tramo 1-3
        data = []
        for r in rows:
            mn = df_params.iat[r, c_min]
            mx = df_params.iat[r, c_max]
            fija = df_params.iat[r, c_fija]
            if pd.isna(mn) or pd.isna(mx) or pd.isna(fija):
                continue
            data.append(
                {
                    "Tramo": f"Tramo {len(data) + 1}",
                    "Min": float(mn),
                    "Max": float(mx),
                    "Fija_USD": float(fija),
                }
            )
        tramos_por_pais[pais] = pd.DataFrame(data)

    return tramos_por_pais


# ---------------------------------------------------------
# Cálculo del Acceso proyectado (por tramos mensuales)
# ---------------------------------------------------------

def calcular_acceso_proyectado_por_tramos(
    df_bbdd: pd.DataFrame, tramos_por_pais: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calcula el Acceso proyectado por corredor usando tramos mensuales.

    Lógica:
      1. Cada fila de A.3 BBDD Neg es (Pais, Año, Mes, Corredor, Monto USD).
      2. Según el Monto USD mensual y los tramos del país, se asigna una tarifa fija mensual.
      3. Se suman las 12 tarifas mensuales por corredor:
         Acceso_Proyectado = suma(Fija_USD_mes_1..12)
    """
    df = df_bbdd.copy()

    # Estandarizamos nombres
    df.rename(
        columns={
            "Pais": "Pais",
            "Corredor": "Broker",
            "Monto USD": "Monto_USD",
        },
        inplace=True,
    )

    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors="coerce").fillna(0.0)
    df["Pais"] = df["Pais"].astype(str).str.strip()

    def fee_for_row(row) -> float:
        pais = row["Pais"]
        vol = row["Monto_USD"]
        tramos_df = tramos_por_pais.get(pais)
        if tramos_df is None or tramos_df.empty:
            return 0.0
        for _, t in tramos_df.iterrows():
            if vol >= t["Min"] and vol <= t["Max"]:
                return float(t["Fija_USD"])
        return 0.0

    # Tarifa mensual por fila
    df["Acceso_Proyectado_mensual"] = df.apply(fee_for_row, axis=1)

    # Agregamos a nivel corredor (anual)
    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_USD", "Acceso_Proyectado_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(
        columns={
            "Monto_USD": "Monto_USD_Total",
            "Acceso_Proyectado_mensual": "Acceso_Proyectado",
        },
        inplace=True,
    )

    return agg


# ---------------------------------------------------------
# Construcción de "Real" desde 3. Negociación
# ---------------------------------------------------------

def construir_real_desde_negociacion(
    df_neg: pd.DataFrame, df_bbdd: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye una tabla Real por corredor a partir de la hoja 3. Negociación.

    Usa la BBDD para inferir el país de cada corredor (modo de la columna Pais).
    """
    # Mapeo corredor -> país usando la BBDD
    broker_country = (
        df_bbdd.groupby("Corredor")["Pais"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    broker_country.rename(columns={"Corredor": "Broker"}, inplace=True)

    # La columna "Real" de la hoja 3. Negociación es el Ingreso Acceso Real anual
    df_real = df_neg[["Corredor", "Real"]].copy()
    df_real.rename(columns={"Corredor": "Broker", "Real": "Acceso_Real"}, inplace=True)
    df_real["Acceso_Real"] = pd.to_numeric(df_real["Acceso_Real"], errors="coerce").fillna(
        0.0
    )

    df_real = pd.merge(df_real, broker_country, on="Broker", how="left")

    # Quitamos filas sin país (ej. BCS agregado, que ya se reparte en los brokers)
    df_real = df_real[~df_real["Pais"].isna()].copy()

    return df_real


# ---------------------------------------------------------
# Unión de Real y Proyectado + KPIs
# ---------------------------------------------------------

def unir_real_y_proyectado(df_real: pd.DataFrame, df_proj: pd.DataFrame) -> pd.DataFrame:
    """Une tablas Real y Proyectado por Pais/Broker y calcula variación %."""
    df = pd.merge(
        df_real,
        df_proj,
        on=["Pais", "Broker"],
        how="outer",
        suffixes=("_Real", "_Proj"),
    )

    df["Acceso_Real"] = df["Acceso_Real"].fillna(0.0)
    df["Acceso_Proyectado"] = df["Acceso_Proyectado"].fillna(0.0)
    df["Monto_USD_Total"] = df["Monto_USD_Total"].fillna(0.0)

    df["Var_%"] = np.where(
        df["Acceso_Real"] != 0,
        (df["Acceso_Proyectado"] - df["Acceso_Real"]) / df["Acceso_Real"] * 100,
        0.0,
    )

    return df


def kpis_por_pais(df_brokers: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_brokers.groupby("Pais")[["Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado"]]
        .sum()
        .reset_index()
    )
    agg["Var_%_Ingresos"] = np.where(
        agg["Acceso_Real"] != 0,
        (agg["Acceso_Proyectado"] - agg["Acceso_Real"]) / agg["Acceso_Real"] * 100,
        0.0,
    )
    agg["BPS_Acceso_Proyectado"] = np.where(
        agg["Monto_USD_Total"] != 0,
        agg["Acceso_Proyectado"] / agg["Monto_USD_Total"] * 10000,
        0.0,
    )
    return agg


def preparar_tabla_brokers(
    df_brokers: pd.DataFrame, pais: Optional[str] = None
) -> pd.DataFrame:
    df = df_brokers.copy()
    if pais is not None:
        df = df[df["Pais"] == pais]

    df_view = df[
        ["Broker", "Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado", "Var_%"]
    ].copy()

    df_view.rename(
        columns={
            "Broker": "Broker / Corredor",
            "Monto_USD_Total": "Monto negociado (USD)",
            "Acceso_Real": "Ingreso real (Acceso actual)",
            "Acceso_Proyectado": "Ingreso proyectado (Simulado)",
            "Var_%": "Var % Ingreso",
        },
        inplace=True,
    )
    return df_view.sort_values("Monto negociado (USD)", ascending=False)


# ---------------------------------------------------------
# Upload del Excel
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📥 Sube el archivo Excel del modelo (por ejemplo: '23102025 Modelamiento Estructura Tarifaria RV (1).xlsx')",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("Sube el Excel para empezar a simular.")
    st.stop()

excel_bytes = uploaded_file.getvalue()

with st.spinner("Cargando hojas del Excel..."):
    try:
        df_bbdd = load_bbdd_from_bytes(excel_bytes, sheet_name="A.3 BBDD Neg")
        df_params = load_param_sheet(excel_bytes)
        df_neg = load_negociacion_sheet(excel_bytes)
    except Exception as e:
        st.error(
            f"No se pudo leer el Excel. Revisa el formato o el nombre de las hojas.\n\nError: {e}"
        )
        st.stop()

st.success("Excel cargado correctamente. Usando 'A.3 BBDD Neg' como BBDD base.")


# ---------------------------------------------------------
# Tramos de acceso (editable en la app)
# ---------------------------------------------------------

st.subheader("🧩 Tramos de Acceso mensual por país")

tramos_por_pais = get_tramos_acceso(df_params)

tab_ch, tab_co, tab_pe = st.tabs(["Chile", "Colombia", "Perú"])

with tab_ch:
    st.markdown("**Chile – Tramos mensuales (USD)**")
    tramos_ch = st.data_editor(
        tramos_por_pais["Chile"],
        num_rows="fixed",
        use_container_width=True,
        key="tramos_chile",
    )

with tab_co:
    st.markdown("**Colombia – Tramos mensuales (USD)**")
    tramos_co = st.data_editor(
        tramos_por_pais["Colombia"],
        num_rows="fixed",
        use_container_width=True,
        key="tramos_colombia",
    )

with tab_pe:
    st.markdown("**Perú – Tramos mensuales (USD)**")
    tramos_pe = st.data_editor(
        tramos_por_pais["Perú"],
        num_rows="fixed",
        use_container_width=True,
        key="tramos_peru",
    )

# Aplicamos posibles cambios del usuario
tramos_por_pais["Chile"] = tramos_ch
tramos_por_pais["Colombia"] = tramos_co
tramos_por_pais["Perú"] = tramos_pe

st.markdown(
    "Los tramos se interpretan así: si en un **mes** el volumen (Monto USD) del corredor cae entre "
    "`Min` y `Max`, se le cobra la `Fija_USD` indicada para ese mes. "
    "El anual es la suma de los 12 meses."
)
st.markdown("---")


# ---------------------------------------------------------
# Cálculo del escenario (Real vs Proyectado)
# ---------------------------------------------------------

with st.spinner("Calculando ingresos de acceso proyectados por corredor..."):
    df_proj = calcular_acceso_proyectado_por_tramos(df_bbdd, tramos_por_pais)
    df_real = construir_real_desde_negociacion(df_neg, df_bbdd)
    df_brokers = unir_real_y_proyectado(df_real, df_proj)
    df_kpis = kpis_por_pais(df_brokers)


# ---------------------------------------------------------
# KPIs por país
# ---------------------------------------------------------

st.subheader("🏁 KPIs de Acceso por País")

# Nos aseguramos de que aparezcan siempre los 3 países
for p in ["Chile", "Colombia", "Perú"]:
    if p not in df_kpis["Pais"].values:
        df_kpis = pd.concat(
            [
                df_kpis,
                pd.DataFrame(
                    {
                        "Pais": [p],
                        "Monto_USD_Total": [0.0],
                        "Acceso_Real": [0.0],
                        "Acceso_Proyectado": [0.0],
                        "Var_%_Ingresos": [0.0],
                        "BPS_Acceso_Proyectado": [0.0],
                    }
                ),
            ],
            ignore_index=True,
        )

df_kpis = df_kpis.set_index("Pais").loc[["Chile", "Colombia", "Perú"]].reset_index()

col1, col2, col3 = st.columns(3)
for col, pais in zip([col1, col2, col3], ["Chile", "Colombia", "Perú"]):
    row = df_kpis[df_kpis["Pais"] == pais].iloc[0]
    col.metric(
        label=f"{pais} – Variación Ingresos Acceso (%)",
        value=f"{row['Var_%_Ingresos']:,.2f} %",
    )
    col.metric(
        label=f"{pais} – BPS Acceso proyectado",
        value=f"{row['BPS_Acceso_Proyectado']:,.1f} bps",
    )

st.markdown("---")


# ---------------------------------------------------------
# Tabla detallada por broker (como 3. Negociación)
# ---------------------------------------------------------

st.subheader("📋 Detalle por Broker")

tab_b_ch, tab_b_co, tab_b_pe, tab_b_all = st.tabs(
    ["Chile", "Colombia", "Perú", "Todos"]
)

with tab_b_ch:
    st.markdown("**Chile – Detalle por broker**")
    st.dataframe(
        preparar_tabla_brokers(df_brokers, "Chile"),
        use_container_width=True,
    )

with tab_b_co:
    st.markdown("**Colombia – Detalle por broker**")
    st.dataframe(
        preparar_tabla_brokers(df_brokers, "Colombia"),
        use_container_width=True,
    )

with tab_b_pe:
    st.markdown("**Perú – Detalle por broker**")
    st.dataframe(
        preparar_tabla_brokers(df_brokers, "Perú"),
        use_container_width=True,
    )

with tab_b_all:
    st.markdown("**Todos los países – Detalle consolidado**")
    st.dataframe(
        preparar_tabla_brokers(df_brokers, None),
        use_container_width=True,
    )

st.markdown("---")


# ---------------------------------------------------------
# Gráfico de barras: Actual vs Proyectado
# ---------------------------------------------------------

st.subheader("📈 Ingreso de Acceso: Actual vs Proyectado por País")

df_chart = df_kpis[["Pais", "Acceso_Real", "Acceso_Proyectado"]].set_index("Pais")
st.bar_chart(df_chart)

st.markdown("---")

st.caption(
    "La lógica replica la hoja '3. Negociación': para cada mes y corredor se asigna una tarifa fija "
    "según tramos de volumen mensual y se suma en el año. "
    "Puedes modificar los tramos (Min / Max / Fija_USD) para simular nuevas estructuras "
    "con 500 / 1,500 / 3,000 u otros valores."
)
