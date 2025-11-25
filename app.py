import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from typing import Optional, Dict
import copy


# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Tarifas RV - nuam",
    layout="wide",
)

st.title("📊 Simulador de Tarifas RV - nuam")
st.caption(
    "Simula las tarifas de **Acceso** y **Transacción** (por tramos mensuales) para Chile, Colombia y Perú "
    "a partir del modelo Excel. Compara ingreso real vs proyectado por broker, por país y entre escenarios."
)


# ---------------------------------------------------------
# Funciones auxiliares de carga
# ---------------------------------------------------------

def load_bbdd_from_bytes(data: bytes, sheet_name: str = "A.3 BBDD Neg") -> pd.DataFrame:
    """Carga la hoja A.3 BBDD Neg. El encabezado está en la fila 7 (index 6)."""
    df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=6)
    return df


def load_param_sheet(excel_bytes: bytes) -> pd.DataFrame:
    """Hoja 1. Parametros sin encabezados."""
    return pd.read_excel(BytesIO(excel_bytes), sheet_name="1. Parametros", header=None)


def load_negociacion_sheet(excel_bytes: bytes) -> pd.DataFrame:
    """Hoja 3. Negociación. El header útil empieza en la fila 9 (index 8)."""
    df = pd.read_excel(BytesIO(excel_bytes), sheet_name="3. Negociación", header=8)
    return df


# ---------------------------------------------------------
# Tramos de ACCESO desde 1. Parametros
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
        rows = [90, 91, 92]
        data = []
        for r in rows:
            mn = df_params.iat[r, c_min]
            mx = df_params.iat[r, c_max]
            fija = df_params.iat[r, c_fija]
            if pd.isna(mn) or pd.isna(mx) or pd.isna(fija):
                continue
            data.append({
                "Tramo": f"Tramo {len(data) + 1}",
                "Min": float(mn),
                "Max": float(mx),
                "Fija_USD": float(fija),
            })
        tramos_por_pais[pais] = pd.DataFrame(data)

    return tramos_por_pais


# ---------------------------------------------------------
# Tramos de TRANSACCIÓN desde 1. Parametros
# ---------------------------------------------------------

def get_tramos_transaccion(df_params: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Lee los tramos de Transacción desde 1. Parametros.
    Ubicación: Filas 129-131 (Tramos 1-3), columnas:
        - Colombia: Min = col 19, Max = col 20, BPS = col 21
        - Perú:     Min = col 23, Max = col 24, BPS = col 25
        - Chile:    Min = col 27, Max = col 28, BPS = col 29
    """
    blocks = {
        "Colombia": (19, 20, 21),
        "Perú": (23, 24, 25),
        "Chile": (27, 28, 29),
    }

    tramos_por_pais: Dict[str, pd.DataFrame] = {}

    for pais, (c_min, c_max, c_bps) in blocks.items():
        rows = [129, 130, 131]
        data = []
        for r in rows:
            mn = df_params.iat[r, c_min]
            mx = df_params.iat[r, c_max]
            bps = df_params.iat[r, c_bps]

            if isinstance(mx, str):
                mx = 9_999_999_999_999.0

            if pd.isna(mn) or pd.isna(mx) or pd.isna(bps):
                continue

            data.append({
                "Tramo": f"Tramo {len(data) + 1}",
                "Min": float(mn),
                "Max": float(mx),
                "BPS": float(bps) * 10_000,
                "Tasa": float(bps),
            })
        tramos_por_pais[pais] = pd.DataFrame(data)

    return tramos_por_pais


# ---------------------------------------------------------
# Cálculo ACCESO proyectado
# ---------------------------------------------------------

def calcular_acceso_proyectado_por_tramos(
    df_bbdd: pd.DataFrame,
    tramos_por_pais: Dict[str, pd.DataFrame],
    scope_chile: str = "DMA",
) -> pd.DataFrame:
    """
    Calcula el Acceso proyectado por corredor usando tramos mensuales.
    
    Para Chile:
      - scope='DMA': usa columna 'Monto DMA USD'
      - scope='TOTAL': usa columna 'Monto USD'
    Para otros países: siempre usa 'Monto USD'
    """
    df = df_bbdd.copy()
    df["Pais"] = df["Pais"].astype(str).str.strip()
    df["Broker"] = df["Corredor"]
    df["Monto_USD"] = pd.to_numeric(df["Monto USD"], errors="coerce").fillna(0.0)
    df["Monto_DMA_USD"] = pd.to_numeric(df.get("Monto DMA USD", 0), errors="coerce").fillna(0.0)

    # Determinar qué monto usar según país y scope
    def get_monto_efectivo(row):
        if row["Pais"] == "Chile" and scope_chile == "DMA":
            return row["Monto_DMA_USD"]
        return row["Monto_USD"]
    
    df["Monto_Efectivo"] = df.apply(get_monto_efectivo, axis=1)

    def fee_for_row(row) -> float:
        pais = row["Pais"]
        vol = row["Monto_Efectivo"]
        tramos_df = tramos_por_pais.get(pais)
        if tramos_df is None or tramos_df.empty:
            return 0.0
        for _, t in tramos_df.iterrows():
            if vol >= t["Min"] and vol <= t["Max"]:
                return float(t["Fija_USD"])
        return 0.0

    df["Acceso_Proyectado_mensual"] = df.apply(fee_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_Efectivo", "Acceso_Proyectado_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(
        columns={
            "Monto_Efectivo": "Monto_USD_Total",
            "Acceso_Proyectado_mensual": "Acceso_Proyectado",
        },
        inplace=True,
    )

    return agg


# ---------------------------------------------------------
# Cálculo TRANSACCIÓN proyectada
# ---------------------------------------------------------

def calcular_transaccion_progresiva_mensual(monto: float, tramos_df: pd.DataFrame) -> float:
    """
    Calcula la tarifa de transacción usando tramos PROGRESIVOS.
    """
    if tramos_df is None or tramos_df.empty or monto <= 0:
        return 0.0

    tarifa_total = 0.0
    tramos_sorted = tramos_df.sort_values("Min").reset_index(drop=True)

    for _, tramo in tramos_sorted.iterrows():
        tramo_min = tramo["Min"]
        tramo_max = tramo["Max"]
        tasa = tramo["Tasa"]

        if monto <= tramo_min:
            continue

        monto_en_tramo = min(monto, tramo_max) - tramo_min
        if monto_en_tramo <= 0:
            continue

        tarifa_total += monto_en_tramo * tasa

    return tarifa_total


def calcular_transaccion_proyectada_por_tramos(
    df_bbdd: pd.DataFrame,
    tramos_por_pais: Dict[str, pd.DataFrame],
    scope_chile: str = "DMA",
) -> pd.DataFrame:
    """
    Calcula la Transacción proyectada por corredor usando tramos progresivos mensuales.
    """
    df = df_bbdd.copy()
    df["Pais"] = df["Pais"].astype(str).str.strip()
    df["Broker"] = df["Corredor"]
    df["Monto_USD"] = pd.to_numeric(df["Monto USD"], errors="coerce").fillna(0.0)
    df["Monto_DMA_USD"] = pd.to_numeric(df.get("Monto DMA USD", 0), errors="coerce").fillna(0.0)

    def get_monto_efectivo(row):
        if row["Pais"] == "Chile" and scope_chile == "DMA":
            return row["Monto_DMA_USD"]
        return row["Monto_USD"]
    
    df["Monto_Efectivo"] = df.apply(get_monto_efectivo, axis=1)

    def transaccion_for_row(row) -> float:
        pais = row["Pais"]
        monto = row["Monto_Efectivo"]
        tramos_df = tramos_por_pais.get(pais)
        return calcular_transaccion_progresiva_mensual(monto, tramos_df)

    df["Transaccion_Proyectada_mensual"] = df.apply(transaccion_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_Efectivo", "Transaccion_Proyectada_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(
        columns={
            "Monto_Efectivo": "Monto_USD_Total",
            "Transaccion_Proyectada_mensual": "Transaccion_Proyectada",
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
    """Construye tabla de Acceso Real y Transacción Real por corredor."""
    broker_country = (
        df_bbdd.groupby("Corredor")["Pais"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    broker_country.rename(columns={"Corredor": "Broker"}, inplace=True)

    df_real = df_neg[["Corredor", "Real", "Real.1"]].copy()
    df_real.rename(
        columns={
            "Corredor": "Broker",
            "Real": "Acceso_Real",
            "Real.1": "Transaccion_Real",
        },
        inplace=True,
    )

    df_real["Acceso_Real"] = pd.to_numeric(df_real["Acceso_Real"], errors="coerce").fillna(0.0)
    df_real["Transaccion_Real"] = pd.to_numeric(df_real["Transaccion_Real"], errors="coerce").fillna(0.0)

    df_real = pd.merge(df_real, broker_country, on="Broker", how="left")
    df_real = df_real[~df_real["Pais"].isna()].copy()

    return df_real


# ---------------------------------------------------------
# Unión de Real y Proyectado + KPIs
# ---------------------------------------------------------

def unir_real_y_proyectado(
    df_real: pd.DataFrame,
    df_proj_acceso: pd.DataFrame,
    df_proj_trans: pd.DataFrame,
) -> pd.DataFrame:
    """Une tablas Real y Proyectado (Acceso + Transacción) por Pais/Broker."""
    df = pd.merge(
        df_real,
        df_proj_acceso[["Pais", "Broker", "Monto_USD_Total", "Acceso_Proyectado"]],
        on=["Pais", "Broker"],
        how="outer",
    )

    df = pd.merge(
        df,
        df_proj_trans[["Pais", "Broker", "Transaccion_Proyectada"]],
        on=["Pais", "Broker"],
        how="outer",
    )

    for col in ["Acceso_Real", "Acceso_Proyectado", "Transaccion_Real",
                "Transaccion_Proyectada", "Monto_USD_Total"]:
        df[col] = df[col].fillna(0.0)

    df["Var_%_Acceso"] = np.where(
        df["Acceso_Real"] != 0,
        (df["Acceso_Proyectado"] - df["Acceso_Real"]) / df["Acceso_Real"] * 100,
        0.0,
    )

    df["Var_%_Transaccion"] = np.where(
        df["Transaccion_Real"] != 0,
        (df["Transaccion_Proyectada"] - df["Transaccion_Real"]) / df["Transaccion_Real"] * 100,
        0.0,
    )

    df["Ingreso_Real_Total"] = df["Acceso_Real"] + df["Transaccion_Real"]
    df["Ingreso_Proyectado_Total"] = df["Acceso_Proyectado"] + df["Transaccion_Proyectada"]

    df["Var_%_Total"] = np.where(
        df["Ingreso_Real_Total"] != 0,
        (df["Ingreso_Proyectado_Total"] - df["Ingreso_Real_Total"]) / df["Ingreso_Real_Total"] * 100,
        0.0,
    )

    return df


def kpis_por_pais(df_brokers: pd.DataFrame) -> pd.DataFrame:
    """Calcula KPIs agregados por país para Acceso y Transacción."""
    agg = (
        df_brokers.groupby("Pais")[
            ["Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado",
             "Transaccion_Real", "Transaccion_Proyectada"]
        ]
        .sum()
        .reset_index()
    )

    agg["Var_%_Acceso"] = np.where(
        agg["Acceso_Real"] != 0,
        (agg["Acceso_Proyectado"] - agg["Acceso_Real"]) / agg["Acceso_Real"] * 100,
        0.0,
    )

    agg["Var_%_Transaccion"] = np.where(
        agg["Transaccion_Real"] != 0,
        (agg["Transaccion_Proyectada"] - agg["Transaccion_Real"]) / agg["Transaccion_Real"] * 100,
        0.0,
    )

    agg["Ingreso_Real_Total"] = agg["Acceso_Real"] + agg["Transaccion_Real"]
    agg["Ingreso_Proyectado_Total"] = agg["Acceso_Proyectado"] + agg["Transaccion_Proyectada"]

    agg["Var_%_Total"] = np.where(
        agg["Ingreso_Real_Total"] != 0,
        (agg["Ingreso_Proyectado_Total"] - agg["Ingreso_Real_Total"]) / agg["Ingreso_Real_Total"] * 100,
        0.0,
    )

    agg["BPS_Acceso_Proyectado"] = np.where(
        agg["Monto_USD_Total"] != 0,
        agg["Acceso_Proyectado"] / agg["Monto_USD_Total"] * 10_000,
        0.0,
    )

    agg["BPS_Transaccion_Proyectada"] = np.where(
        agg["Monto_USD_Total"] != 0,
        agg["Transaccion_Proyectada"] / agg["Monto_USD_Total"] * 10_000,
        0.0,
    )

    agg["BPS_Total_Proyectado"] = agg["BPS_Acceso_Proyectado"] + agg["BPS_Transaccion_Proyectada"]

    return agg


def preparar_tabla_brokers(
    df_brokers: pd.DataFrame, pais: Optional[str] = None, concepto: str = "Total"
) -> pd.DataFrame:
    """Prepara tabla de brokers filtrada por país y concepto."""
    df = df_brokers.copy()
    if pais is not None:
        df = df[df["Pais"] == pais]

    if concepto == "Acceso":
        df_view = df[["Broker", "Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado", "Var_%_Acceso"]].copy()
        df_view.columns = ["Broker / Corredor", "Monto negociado (USD)", "Acceso Real", "Acceso Proyectado", "Var % Acceso"]
    elif concepto == "Transaccion":
        df_view = df[["Broker", "Monto_USD_Total", "Transaccion_Real", "Transaccion_Proyectada", "Var_%_Transaccion"]].copy()
        df_view.columns = ["Broker / Corredor", "Monto negociado (USD)", "Transacción Real", "Transacción Proyectada", "Var % Transacción"]
    else:
        df_view = df[["Broker", "Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado",
                      "Transaccion_Real", "Transaccion_Proyectada",
                      "Ingreso_Real_Total", "Ingreso_Proyectado_Total", "Var_%_Total"]].copy()
        df_view.columns = ["Broker / Corredor", "Monto (USD)", "Acceso Real", "Acceso Proy.",
                          "Trans. Real", "Trans. Proy.", "Total Real", "Total Proy.", "Var % Total"]

    return df_view.sort_values(df_view.columns[1], ascending=False)


# ---------------------------------------------------------
# Función para copiar escenarios de forma segura (deep copy)
# ---------------------------------------------------------

def copiar_escenario(params: Dict) -> Dict:
    """Crea una copia profunda de los parámetros de un escenario."""
    return {
        "tramos_acceso": {k: v.copy() for k, v in params["tramos_acceso"].items()},
        "tramos_transaccion": {k: v.copy() for k, v in params["tramos_transaccion"].items()},
        "scope_chile": params.get("scope_chile", "DMA"),
    }


# ---------------------------------------------------------
# Simulación completa de un escenario
# ---------------------------------------------------------

def simular_escenario(
    params: Dict,
    df_bbdd: pd.DataFrame,
    df_neg: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame):
    """Ejecuta la simulación completa de un escenario."""
    tr_acc = params["tramos_acceso"]
    tr_tr = params["tramos_transaccion"]
    scope_ch = params.get("scope_chile", "DMA")

    df_proj_acc = calcular_acceso_proyectado_por_tramos(df_bbdd, tr_acc, scope_ch)
    df_proj_tr = calcular_transaccion_proyectada_por_tramos(df_bbdd, tr_tr, scope_ch)
    df_real = construir_real_desde_negociacion(df_neg, df_bbdd)
    df_brokers = unir_real_y_proyectado(df_real, df_proj_acc, df_proj_tr)
    df_kpis = kpis_por_pais(df_brokers)

    # Asegurar Chile, Colombia, Perú siempre presentes
    for p in ["Chile", "Colombia", "Perú"]:
        if p not in df_kpis["Pais"].values:
            df_kpis = pd.concat([
                df_kpis,
                pd.DataFrame({
                    "Pais": [p],
                    "Monto_USD_Total": [0.0],
                    "Acceso_Real": [0.0], "Acceso_Proyectado": [0.0],
                    "Transaccion_Real": [0.0], "Transaccion_Proyectada": [0.0],
                    "Var_%_Acceso": [0.0], "Var_%_Transaccion": [0.0], "Var_%_Total": [0.0],
                    "Ingreso_Real_Total": [0.0], "Ingreso_Proyectado_Total": [0.0],
                    "BPS_Acceso_Proyectado": [0.0], "BPS_Transaccion_Proyectada": [0.0],
                    "BPS_Total_Proyectado": [0.0],
                }),
            ], ignore_index=True)

    df_kpis = df_kpis.set_index("Pais").loc[["Chile", "Colombia", "Perú"]].reset_index()
    return df_brokers, df_kpis


# ---------------------------------------------------------
# Upload del Excel
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📥 Sube el archivo Excel del modelo",
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
        st.error(f"No se pudo leer el Excel.\n\nError: {e}")
        st.stop()

st.success("✅ Excel cargado correctamente")


# ---------------------------------------------------------
# Inicialización de estado
# ---------------------------------------------------------

base_tramos_acceso = get_tramos_acceso(df_params)
base_tramos_trans = get_tramos_transaccion(df_params)

if "current_params" not in st.session_state:
    st.session_state["current_params"] = {
        "tramos_acceso": {k: v.copy() for k, v in base_tramos_acceso.items()},
        "tramos_transaccion": {k: v.copy() for k, v in base_tramos_trans.items()},
        "scope_chile": "DMA",
    }

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

if "current_scenario_name" not in st.session_state:
    st.session_state["current_scenario_name"] = "Escenario actual"


# ---------------------------------------------------------
# SIDEBAR: Gestión de escenarios
# ---------------------------------------------------------

st.sidebar.header("📁 Gestión de Escenarios")

# Mostrar escenario activo
st.sidebar.markdown(f"**Escenario activo:** `{st.session_state['current_scenario_name']}`")
st.sidebar.markdown(f"**Scope Chile:** `{st.session_state['current_params'].get('scope_chile', 'DMA')}`")

st.sidebar.markdown("---")

# Guardar escenario
st.sidebar.subheader("💾 Guardar escenario")
nuevo_nombre = st.sidebar.text_input("Nombre:", value="", key="input_nuevo_nombre")

if st.sidebar.button("💾 Guardar", use_container_width=True):
    if nuevo_nombre.strip():
        st.session_state["scenarios"][nuevo_nombre.strip()] = copiar_escenario(st.session_state["current_params"])
        st.session_state["current_scenario_name"] = nuevo_nombre.strip()
        st.sidebar.success(f"✅ '{nuevo_nombre}' guardado")
    else:
        st.sidebar.warning("⚠️ Escribe un nombre")

st.sidebar.markdown("---")

# Cargar escenario
st.sidebar.subheader("📂 Cargar escenario")
escenarios_guardados = list(st.session_state["scenarios"].keys())

if escenarios_guardados:
    escenario_a_cargar = st.sidebar.selectbox(
        "Selecciona:",
        options=escenarios_guardados,
        key="select_cargar"
    )
    
    col_load, col_del = st.sidebar.columns(2)
    
    with col_load:
        if st.button("📂 Cargar", use_container_width=True):
            st.session_state["current_params"] = copiar_escenario(st.session_state["scenarios"][escenario_a_cargar])
            st.session_state["current_scenario_name"] = escenario_a_cargar
            st.rerun()
    
    with col_del:
        if st.button("🗑️ Borrar", use_container_width=True):
            del st.session_state["scenarios"][escenario_a_cargar]
            st.sidebar.success(f"🗑️ '{escenario_a_cargar}' eliminado")
            st.rerun()
else:
    st.sidebar.info("No hay escenarios guardados")

st.sidebar.markdown("---")

# Lista de escenarios
if escenarios_guardados:
    st.sidebar.subheader("📋 Escenarios guardados")
    for nombre in escenarios_guardados:
        scope = st.session_state["scenarios"][nombre].get("scope_chile", "DMA")
        st.sidebar.markdown(f"- `{nombre}` (Chile: {scope})")


# ---------------------------------------------------------
# Referencias a parámetros actuales
# ---------------------------------------------------------

current_params = st.session_state["current_params"]
tramos_acceso = current_params["tramos_acceso"]
tramos_transaccion = current_params["tramos_transaccion"]
scope_chile = current_params.get("scope_chile", "DMA")


# ---------------------------------------------------------
# TABS principales
# ---------------------------------------------------------

tab_config, tab_kpis, tab_comparar, tab_brokers = st.tabs([
    "⚙️ Configuración", 
    "📊 KPIs por País", 
    "⚔️ Comparar Escenarios",
    "📋 Detalle Brokers"
])


# ---------------------------------------------------------
# TAB 1: Configuración
# ---------------------------------------------------------

with tab_config:
    st.subheader("⚙️ Configuración del Escenario Actual")
    
    # Indicador visual del escenario activo
    st.info(f"📌 **Escenario activo:** {st.session_state['current_scenario_name']} | **Scope Chile:** {scope_chile}")
    
    # Alcance Chile
    st.markdown("### 🇨🇱 Alcance de cobro en Chile")
    
    scope_options = ["Solo DMA (67% del volumen)", "Todo el volumen (100%)"]
    scope_index = 0 if scope_chile == "DMA" else 1
    
    scope_label = st.radio(
        "¿Sobre qué volumen se aplican las tarifas en Chile?",
        scope_options,
        index=scope_index,
        horizontal=True,
        key="scope_radio"
    )
    
    new_scope = "DMA" if "DMA" in scope_label else "TOTAL"
    if new_scope != scope_chile:
        st.session_state["current_params"]["scope_chile"] = new_scope
        st.rerun()
    
    # Mostrar estadísticas de volumen
    monto_total_chile = df_bbdd[df_bbdd["Pais"] == "Chile"]["Monto USD"].sum()
    monto_dma_chile = df_bbdd[df_bbdd["Pais"] == "Chile"]["Monto DMA USD"].sum()
    
    col_vol1, col_vol2, col_vol3 = st.columns(3)
    col_vol1.metric("Volumen Total Chile", f"${monto_total_chile/1e9:.2f}B")
    col_vol2.metric("Volumen DMA Chile", f"${monto_dma_chile/1e9:.2f}B")
    col_vol3.metric("% DMA", f"{monto_dma_chile/monto_total_chile*100:.1f}%")
    
    st.markdown("---")
    
    # Tramos de Acceso
    st.markdown("### 🔐 Tramos de ACCESO mensual por país")
    
    tab_acc_ch, tab_acc_co, tab_acc_pe = st.tabs(["Chile", "Colombia", "Perú"])
    
    with tab_acc_ch:
        tramos_acc_ch = st.data_editor(
            tramos_acceso["Chile"],
            num_rows="fixed",
            use_container_width=True,
            key="acc_chile",
        )
        st.session_state["current_params"]["tramos_acceso"]["Chile"] = tramos_acc_ch
    
    with tab_acc_co:
        tramos_acc_co = st.data_editor(
            tramos_acceso["Colombia"],
            num_rows="fixed",
            use_container_width=True,
            key="acc_colombia",
        )
        st.session_state["current_params"]["tramos_acceso"]["Colombia"] = tramos_acc_co
    
    with tab_acc_pe:
        tramos_acc_pe = st.data_editor(
            tramos_acceso["Perú"],
            num_rows="fixed",
            use_container_width=True,
            key="acc_peru",
        )
        st.session_state["current_params"]["tramos_acceso"]["Perú"] = tramos_acc_pe
    
    st.markdown("---")
    
    # Tramos de Transacción
    st.markdown("### 💱 Tramos de TRANSACCIÓN mensual por país")
    st.caption("Sistema progresivo: cada tramo aplica solo al monto dentro de su rango")
    
    tab_tr_ch, tab_tr_co, tab_tr_pe = st.tabs(["Chile", "Colombia", "Perú"])
    
    def actualizar_tramos_trans(df_edit: pd.DataFrame) -> pd.DataFrame:
        df = df_edit.copy()
        df["Tasa"] = df["BPS"] / 10_000
        return df
    
    with tab_tr_ch:
        tramos_tr_ch = st.data_editor(
            tramos_transaccion["Chile"][["Tramo", "Min", "Max", "BPS"]],
            num_rows="fixed",
            use_container_width=True,
            key="tr_chile",
        )
        st.session_state["current_params"]["tramos_transaccion"]["Chile"] = actualizar_tramos_trans(tramos_tr_ch)
    
    with tab_tr_co:
        tramos_tr_co = st.data_editor(
            tramos_transaccion["Colombia"][["Tramo", "Min", "Max", "BPS"]],
            num_rows="fixed",
            use_container_width=True,
            key="tr_colombia",
        )
        st.session_state["current_params"]["tramos_transaccion"]["Colombia"] = actualizar_tramos_trans(tramos_tr_co)
    
    with tab_tr_pe:
        tramos_tr_pe = st.data_editor(
            tramos_transaccion["Perú"][["Tramo", "Min", "Max", "BPS"]],
            num_rows="fixed",
            use_container_width=True,
            key="tr_peru",
        )
        st.session_state["current_params"]["tramos_transaccion"]["Perú"] = actualizar_tramos_trans(tramos_tr_pe)


# ---------------------------------------------------------
# Cálculo del escenario actual
# ---------------------------------------------------------

df_brokers, df_kpis = simular_escenario(st.session_state["current_params"], df_bbdd, df_neg)


# ---------------------------------------------------------
# TAB 2: KPIs por País
# ---------------------------------------------------------

with tab_kpis:
    st.subheader("📊 KPIs por País – Escenario Actual")
    
    st.info(f"📌 **{st.session_state['current_scenario_name']}** | Scope Chile: {st.session_state['current_params'].get('scope_chile', 'DMA')}")
    
    col1, col2, col3 = st.columns(3)
    
    for col, pais in zip([col1, col2, col3], ["Chile", "Colombia", "Perú"]):
        row = df_kpis[df_kpis["Pais"] == pais].iloc[0]
        col.markdown(f"### {pais}")
        
        # Variaciones con color
        var_total = row['Var_%_Total']
        delta_color = "normal" if var_total >= 0 else "inverse"
        
        col.metric("Ingreso Proyectado", f"${row['Ingreso_Proyectado_Total']:,.0f}", 
                   f"{var_total:+.1f}% vs Real", delta_color=delta_color)
        col.metric("Var % Acceso", f"{row['Var_%_Acceso']:+.1f}%")
        col.metric("Var % Transacción", f"{row['Var_%_Transaccion']:+.1f}%")
        col.metric("BPS Total", f"{row['BPS_Total_Proyectado']:.2f}")
    
    st.markdown("---")
    
    # Tabla resumen
    st.subheader("📊 Resumen de Ingresos")
    
    df_resumen = df_kpis[[
        "Pais", "Monto_USD_Total",
        "Acceso_Real", "Acceso_Proyectado", "Var_%_Acceso",
        "Transaccion_Real", "Transaccion_Proyectada", "Var_%_Transaccion",
        "Ingreso_Real_Total", "Ingreso_Proyectado_Total", "Var_%_Total",
    ]].copy()
    
    df_resumen.columns = ["País", "Monto (USD)", "Acceso Real", "Acceso Proy.", "Var% Acc.",
                          "Trans. Real", "Trans. Proy.", "Var% Trans.",
                          "Total Real", "Total Proy.", "Var% Total"]
    
    st.dataframe(
        df_resumen.style.format({
            "Monto (USD)": "${:,.0f}",
            "Acceso Real": "${:,.0f}",
            "Acceso Proy.": "${:,.0f}",
            "Var% Acc.": "{:+.1f}%",
            "Trans. Real": "${:,.0f}",
            "Trans. Proy.": "${:,.0f}",
            "Var% Trans.": "{:+.1f}%",
            "Total Real": "${:,.0f}",
            "Total Proy.": "${:,.0f}",
            "Var% Total": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    # Gráfico
    st.subheader("📈 Real vs Proyectado")
    
    df_chart = df_kpis[["Pais", "Ingreso_Real_Total", "Ingreso_Proyectado_Total"]].copy()
    df_chart_long = df_chart.melt(id_vars="Pais", var_name="Tipo", value_name="Valor")
    df_chart_long["Tipo"] = df_chart_long["Tipo"].replace({
        "Ingreso_Real_Total": "Real",
        "Ingreso_Proyectado_Total": "Proyectado"
    })
    
    chart = (
        alt.Chart(df_chart_long)
        .mark_bar()
        .encode(
            x=alt.X("Pais:N", title="País"),
            xOffset="Tipo:N",
            y=alt.Y("Valor:Q", axis=alt.Axis(format=",.0f", title="USD")),
            color=alt.Color("Tipo:N", title=""),
            tooltip=["Pais", "Tipo", alt.Tooltip("Valor:Q", format=",.0f")],
        )
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: Comparar Escenarios
# ---------------------------------------------------------

with tab_comparar:
    st.subheader("⚔️ Comparación de Escenarios")
    
    # Opciones: escenario actual + guardados
    opciones_escenarios = ["📌 Escenario actual"] + [f"💾 {k}" for k in st.session_state["scenarios"].keys()]
    
    if len(opciones_escenarios) < 2:
        st.warning("⚠️ Guarda al menos un escenario para poder comparar.")
        st.markdown("""
        **Cómo usar:**
        1. Configura los tramos y el scope de Chile en la pestaña "Configuración"
        2. Dale un nombre en el sidebar y haz clic en "Guardar"
        3. Modifica la configuración y guarda otro escenario
        4. Vuelve aquí para comparar
        """)
    else:
        col_a, col_b = st.columns(2)
        
        with col_a:
            sel_a = st.selectbox("🅰️ Escenario A:", opciones_escenarios, index=0, key="comp_a")
        with col_b:
            sel_b = st.selectbox("🅱️ Escenario B:", opciones_escenarios, 
                                index=min(1, len(opciones_escenarios)-1), key="comp_b")
        
        # Obtener parámetros
        def get_params_from_selection(sel):
            if sel == "📌 Escenario actual":
                return st.session_state["current_params"], "Escenario actual"
            else:
                nombre = sel.replace("💾 ", "")
                return st.session_state["scenarios"][nombre], nombre
        
        params_a, nombre_a = get_params_from_selection(sel_a)
        params_b, nombre_b = get_params_from_selection(sel_b)
        
        # Validar que no sean el mismo
        if sel_a == sel_b:
            st.warning("⚠️ Selecciona dos escenarios diferentes para comparar")
        else:
            # Simular ambos
            df_brok_a, df_kpis_a = simular_escenario(params_a, df_bbdd, df_neg)
            df_brok_b, df_kpis_b = simular_escenario(params_b, df_bbdd, df_neg)
            
            # Mostrar configuración de cada escenario
            st.markdown("### 📋 Configuración de escenarios")
            
            col_conf_a, col_conf_b = st.columns(2)
            
            with col_conf_a:
                st.markdown(f"**🅰️ {nombre_a}**")
                st.markdown(f"- Scope Chile: `{params_a.get('scope_chile', 'DMA')}`")
                # Mostrar tarifas de acceso Chile como ejemplo
                acc_chile_a = params_a["tramos_acceso"]["Chile"]["Fija_USD"].tolist()
                st.markdown(f"- Acceso Chile: {acc_chile_a}")
            
            with col_conf_b:
                st.markdown(f"**🅱️ {nombre_b}**")
                st.markdown(f"- Scope Chile: `{params_b.get('scope_chile', 'DMA')}`")
                acc_chile_b = params_b["tramos_acceso"]["Chile"]["Fija_USD"].tolist()
                st.markdown(f"- Acceso Chile: {acc_chile_b}")
            
            st.markdown("---")
            
            # Tabla comparativa
            st.markdown("### 📊 Comparación de Resultados")
            
            df_comp = pd.DataFrame({
                "País": ["Chile", "Colombia", "Perú"],
            })
            
            for pais in ["Chile", "Colombia", "Perú"]:
                kpi_a = df_kpis_a[df_kpis_a["Pais"] == pais].iloc[0]
                kpi_b = df_kpis_b[df_kpis_b["Pais"] == pais].iloc[0]
                
                idx = df_comp[df_comp["País"] == pais].index[0]
                df_comp.loc[idx, f"Total {nombre_a}"] = kpi_a["Ingreso_Proyectado_Total"]
                df_comp.loc[idx, f"Total {nombre_b}"] = kpi_b["Ingreso_Proyectado_Total"]
                df_comp.loc[idx, "Δ Absoluto"] = kpi_b["Ingreso_Proyectado_Total"] - kpi_a["Ingreso_Proyectado_Total"]
                df_comp.loc[idx, "Δ %"] = (
                    (kpi_b["Ingreso_Proyectado_Total"] - kpi_a["Ingreso_Proyectado_Total"]) 
                    / kpi_a["Ingreso_Proyectado_Total"] * 100
                ) if kpi_a["Ingreso_Proyectado_Total"] > 0 else 0
            
            # Agregar fila de totales
            total_a = df_comp[f"Total {nombre_a}"].sum()
            total_b = df_comp[f"Total {nombre_b}"].sum()
            
            df_total = pd.DataFrame({
                "País": ["**TOTAL**"],
                f"Total {nombre_a}": [total_a],
                f"Total {nombre_b}": [total_b],
                "Δ Absoluto": [total_b - total_a],
                "Δ %": [(total_b - total_a) / total_a * 100 if total_a > 0 else 0],
            })
            
            df_comp = pd.concat([df_comp, df_total], ignore_index=True)
            
            st.dataframe(
                df_comp.style.format({
                    f"Total {nombre_a}": "${:,.0f}",
                    f"Total {nombre_b}": "${:,.0f}",
                    "Δ Absoluto": "${:+,.0f}",
                    "Δ %": "{:+.1f}%",
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                    subset=["Δ Absoluto", "Δ %"]
                ),
                use_container_width=True,
                hide_index=True,
            )
            
            # Gráfico comparativo
            st.markdown("### 📈 Gráfico Comparativo")
            
            df_chart_comp = df_comp[df_comp["País"] != "**TOTAL**"].melt(
                id_vars="País", 
                value_vars=[f"Total {nombre_a}", f"Total {nombre_b}"],
                var_name="Escenario", 
                value_name="Ingreso"
            )
            
            chart_comp = (
                alt.Chart(df_chart_comp)
                .mark_bar()
                .encode(
                    x=alt.X("País:N", title="País"),
                    xOffset="Escenario:N",
                    y=alt.Y("Ingreso:Q", axis=alt.Axis(format=",.0f", title="Ingreso Proyectado (USD)")),
                    color=alt.Color("Escenario:N"),
                    tooltip=["País", "Escenario", alt.Tooltip("Ingreso:Q", format=",.0f")],
                )
            ).properties(height=400)
            
            st.altair_chart(chart_comp, use_container_width=True)
            
            # Resumen ejecutivo
            st.markdown("### 📝 Resumen Ejecutivo")
            
            diff_total = total_b - total_a
            diff_pct = (diff_total / total_a * 100) if total_a > 0 else 0
            
            if diff_total > 0:
                st.success(f"""
                ✅ **{nombre_b}** genera **${diff_total:,.0f} más** que {nombre_a} ({diff_pct:+.1f}%)
                
                - Total {nombre_a}: ${total_a:,.0f}
                - Total {nombre_b}: ${total_b:,.0f}
                """)
            elif diff_total < 0:
                st.error(f"""
                ⚠️ **{nombre_b}** genera **${abs(diff_total):,.0f} menos** que {nombre_a} ({diff_pct:+.1f}%)
                
                - Total {nombre_a}: ${total_a:,.0f}
                - Total {nombre_b}: ${total_b:,.0f}
                """)
            else:
                st.info(f"""
                🔄 Ambos escenarios generan el mismo ingreso: ${total_a:,.0f}
                """)


# ---------------------------------------------------------
# TAB 4: Detalle Brokers
# ---------------------------------------------------------

with tab_brokers:
    st.subheader("📋 Detalle por Broker – Escenario Actual")
    
    st.info(f"📌 **{st.session_state['current_scenario_name']}**")
    
    col_filt1, col_filt2 = st.columns(2)
    
    with col_filt1:
        pais_filtro = st.selectbox("Filtrar por país:", ["Todos", "Chile", "Colombia", "Perú"])
    with col_filt2:
        concepto_sel = st.selectbox("Concepto:", ["Total", "Acceso", "Transaccion"])
    
    pais_param = None if pais_filtro == "Todos" else pais_filtro
    
    df_tabla = preparar_tabla_brokers(df_brokers, pais_param, concepto_sel)
    
    st.dataframe(
        df_tabla.style.format({
            col: "${:,.0f}" for col in df_tabla.columns if "USD" in col or "Real" in col or "Proy" in col or "Monto" in col
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("---")
    
    # Gráfico de brokers seleccionados
    st.subheader("📊 Comparar Brokers")
    
    lista_brokers = sorted(df_brokers["Broker"].unique())
    brokers_sel = st.multiselect(
        "Selecciona brokers a comparar:",
        options=lista_brokers,
        default=lista_brokers[:5] if len(lista_brokers) >= 5 else lista_brokers,
    )
    
    if brokers_sel:
        df_brok_sel = df_brokers[df_brokers["Broker"].isin(brokers_sel)].copy()
        
        df_brok_chart = df_brok_sel[["Broker", "Ingreso_Real_Total", "Ingreso_Proyectado_Total"]].melt(
            id_vars="Broker",
            var_name="Tipo",
            value_name="Ingreso"
        )
        df_brok_chart["Tipo"] = df_brok_chart["Tipo"].replace({
            "Ingreso_Real_Total": "Real",
            "Ingreso_Proyectado_Total": "Proyectado"
        })
        
        chart_brok = (
            alt.Chart(df_brok_chart)
            .mark_bar()
            .encode(
                x=alt.X("Broker:N", sort=None, title=""),
                xOffset="Tipo:N",
                y=alt.Y("Ingreso:Q", axis=alt.Axis(format=",.0f", title="USD")),
                color="Tipo:N",
                tooltip=["Broker", "Tipo", alt.Tooltip("Ingreso:Q", format=",.0f")],
            )
        ).properties(height=400)
        
        st.altair_chart(chart_brok, use_container_width=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    "**Simulador de Tarifas RV nuam** | "
    "Acceso + Transacción | "
    "Gestión de escenarios y comparación"
)
