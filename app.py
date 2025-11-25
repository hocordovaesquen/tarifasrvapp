import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional, Dict


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
    "a partir del modelo Excel. Compara ingreso real vs proyectado por broker y por país. "
    "Además, puedes guardar y cargar escenarios (parámetros + modo de cobro en Chile)."
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
    """
    Hoja 3. Negociación donde están:
        Corredor / Real / Proyectado / Var% de Acceso y Transacción.
    El header útil empieza en la fila 9 (index 8).
    """
    df = pd.read_excel(BytesIO(excel_bytes), sheet_name="3. Negociación", header=8)
    return df


# ---------------------------------------------------------
# Helper DMA (para Chile)
# ---------------------------------------------------------

def infer_dma_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intenta encontrar una columna relacionada a DMA en la BBDD.
    Busca cualquier columna cuyo nombre contenga 'dma'.
    """
    for col in df.columns:
        if "dma" in str(col).lower():
            return col
    return None


def aplicar_scope_chile(df: pd.DataFrame, scope_chile: str) -> pd.DataFrame:
    """
    Aplica el alcance de cobro en Chile:
      - 'DMA': solo filas de Chile con marca DMA=TRUE
      - 'TOTAL': todo el volumen de Chile

    Si no se encuentra columna DMA, se devuelve el df sin cambios.
    """
    scope = (scope_chile or "").upper()
    if scope not in ("DMA", "TOTAL"):
        return df

    dma_col = infer_dma_column(df)
    if dma_col is None:
        # No se encontró columna DMA; no filtramos nada.
        return df

    df = df.copy()
    mask_chile = df["Pais"].astype(str).str.upper().eq("CHILE")

    col = df[dma_col]

    if col.dtype == object:
        mask_dma = col.astype(str).str.lower().str.contains("dma")
    else:
        mask_dma = col.fillna(0) != 0

    if scope == "DMA":
        # Dejamos solo las filas DMA en Chile; el resto igual.
        return df[~mask_chile | (mask_chile & mask_dma)]
    else:
        # TOTAL: dejamos todo
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

    Sistema PROGRESIVO: cada tramo aplica solo al monto dentro de su rango.
    En el Excel la tasa está en formato decimal (ej. 0.0001 = 1 bps).
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

            # Convertir "en adelante" o similar a un número muy grande
            if isinstance(mx, str):
                mx = 9_999_999_999_999.0

            if pd.isna(mn) or pd.isna(mx) or pd.isna(bps):
                continue

            data.append({
                "Tramo": f"Tramo {len(data) + 1}",
                "Min": float(mn),
                "Max": float(mx),
                "BPS": float(bps) * 10_000,  # para mostrar en bps
                "Tasa": float(bps),          # tasa decimal para el cálculo
            })
        tramos_por_pais[pais] = pd.DataFrame(data)

    return tramos_por_pais


# ---------------------------------------------------------
# Cálculo del ACCESO proyectado (por tramos mensuales)
# ---------------------------------------------------------

def calcular_acceso_proyectado_por_tramos(
    df_bbdd: pd.DataFrame,
    tramos_por_pais: Dict[str, pd.DataFrame],
    scope_chile: str = "DMA",
) -> pd.DataFrame:
    """
    Calcula el Acceso proyectado por corredor usando tramos mensuales.
    Lógica: cada mes se asigna una tarifa fija según el tramo de volumen.
    """
    df = df_bbdd.copy()
    df.rename(columns={"Pais": "Pais", "Corredor": "Broker", "Monto USD": "Monto_USD"}, inplace=True)
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors="coerce").fillna(0.0)
    df["Pais"] = df["Pais"].astype(str).str.strip()

    # Aplicar alcance en Chile (solo DMA vs todo el volumen)
    df = aplicar_scope_chile(df, scope_chile)

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

    df["Acceso_Proyectado_mensual"] = df.apply(fee_for_row, axis=1)

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
# Cálculo de TRANSACCIÓN proyectada (tramos PROGRESIVOS)
# ---------------------------------------------------------

def calcular_transaccion_progresiva_mensual(monto: float, tramos_df: pd.DataFrame) -> float:
    """
    Calcula la tarifa de transacción usando tramos PROGRESIVOS.
    Cada tramo aplica solo al monto que cae dentro de su rango.
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
    El cálculo se hace mes a mes y luego se suma al año.
    """
    df = df_bbdd.copy()
    df.rename(columns={"Pais": "Pais", "Corredor": "Broker", "Monto USD": "Monto_USD"}, inplace=True)
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors="coerce").fillna(0.0)
    df["Pais"] = df["Pais"].astype(str).str.strip()

    # Alcance en Chile (DMA vs Total)
    df = aplicar_scope_chile(df, scope_chile)

    def transaccion_for_row(row) -> float:
        pais = row["Pais"]
        monto = row["Monto_USD"]
        tramos_df = tramos_por_pais.get(pais)
        return calcular_transaccion_progresiva_mensual(monto, tramos_df)

    df["Transaccion_Proyectada_mensual"] = df.apply(transaccion_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_USD", "Transaccion_Proyectada_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(
        columns={
            "Monto_USD": "Monto_USD_Total",
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
    """
    Construye una tabla Real por corredor a partir de la hoja 3. Negociación.
    Incluye Acceso Real y Transacción Real.
    """
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
        df_view.rename(
            columns={
                "Broker": "Broker / Corredor",
                "Monto_USD_Total": "Monto negociado (USD)",
                "Acceso_Real": "Acceso Real",
                "Acceso_Proyectado": "Acceso Proyectado",
                "Var_%_Acceso": "Var % Acceso",
            },
            inplace=True,
        )
    elif concepto == "Transaccion":
        df_view = df[
            ["Broker", "Monto_USD_Total", "Transaccion_Real", "Transaccion_Proyectada", "Var_%_Transaccion"]
        ].copy()
        df_view.rename(
            columns={
                "Broker": "Broker / Corredor",
                "Monto_USD_Total": "Monto negociado (USD)",
                "Transaccion_Real": "Transacción Real",
                "Transaccion_Proyectada": "Transacción Proyectada",
                "Var_%_Transaccion": "Var % Transacción",
            },
            inplace=True,
        )
    else:
        df_view = df[
            ["Broker", "Monto_USD_Total", "Acceso_Real", "Acceso_Proyectado",
             "Transaccion_Real", "Transaccion_Proyectada",
             "Ingreso_Real_Total", "Ingreso_Proyectado_Total", "Var_%_Total"]
        ].copy()
        df_view.rename(
            columns={
                "Broker": "Broker / Corredor",
                "Monto_USD_Total": "Monto (USD)",
                "Acceso_Real": "Acceso Real",
                "Acceso_Proyectado": "Acceso Proy.",
                "Transaccion_Real": "Trans. Real",
                "Transaccion_Proyectada": "Trans. Proy.",
                "Ingreso_Real_Total": "Total Real",
                "Ingreso_Proyectado_Total": "Total Proy.",
                "Var_%_Total": "Var % Total",
            },
            inplace=True,
        )

    return df_view.sort_values(df_view.columns[1], ascending=False)


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
        st.error(f"No se pudo leer el Excel. Revisa el formato o el nombre de las hojas.\n\nError: {e}")
        st.stop()

st.success("✅ Excel cargado correctamente. Usando 'A.3 BBDD Neg' como BBDD base.")


# ---------------------------------------------------------
# Inicialización de estado (escenarios / parámetros actuales)
# ---------------------------------------------------------

base_tramos_acceso = get_tramos_acceso(df_params)
base_tramos_trans = get_tramos_transaccion(df_params)

if "current_params" not in st.session_state:
    st.session_state["current_params"] = {
        "tramos_acceso": {k: v.copy() for k, v in base_tramos_acceso.items()},
        "tramos_transaccion": {k: v.copy() for k, v in base_tramos_trans.items()},
        "scope_chile": "DMA",  # por defecto: como hoy
    }

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

current_params = st.session_state["current_params"]
tramos_acceso = current_params["tramos_acceso"]
tramos_transaccion = current_params["tramos_transaccion"]
scope_chile = current_params.get("scope_chile", "DMA")


# ---------------------------------------------------------
# Tramos de ACCESO (editable en la app)
# ---------------------------------------------------------

st.markdown("---")
st.subheader("🔐 Tramos de ACCESO mensual por país")
st.markdown("*Tarifa fija mensual según el volumen negociado del corredor en ese mes.*")

tab_acc_ch, tab_acc_co, tab_acc_pe = st.tabs(["Chile", "Colombia", "Perú"])

with tab_acc_ch:
    st.markdown("**Chile – Tramos Acceso (USD)**")
    tramos_acc_ch = st.data_editor(
        tramos_acceso["Chile"],
        num_rows="fixed",
        use_container_width=True,
        key="acc_chile",
    )

with tab_acc_co:
    st.markdown("**Colombia – Tramos Acceso (USD)**")
    tramos_acc_co = st.data_editor(
        tramos_acceso["Colombia"],
        num_rows="fixed",
        use_container_width=True,
        key="acc_colombia",
    )

with tab_acc_pe:
    st.markdown("**Perú – Tramos Acceso (USD)**")
    tramos_acc_pe = st.data_editor(
        tramos_acceso["Perú"],
        num_rows="fixed",
        use_container_width=True,
        key="acc_peru",
    )

# Actualizar parámetros actuales
tramos_acceso["Chile"] = tramos_acc_ch
tramos_acceso["Colombia"] = tramos_acc_co
tramos_acceso["Perú"] = tramos_acc_pe
st.session_state["current_params"]["tramos_acceso"] = tramos_acceso


# ---------------------------------------------------------
# Tramos de TRANSACCIÓN (editable en la app)
# ---------------------------------------------------------

st.markdown("---")
st.subheader("💱 Tramos de TRANSACCIÓN mensual por país")
st.markdown(
    "*Sistema **progresivo**: cada tramo aplica solo al monto dentro de su rango. "
    "Por ejemplo: los primeros 100M se cobran al BPS del Tramo 1, de 100M a 300M al BPS del Tramo 2, etc.*"
)

tab_tr_ch, tab_tr_co, tab_tr_pe = st.tabs(["Chile", "Colombia", "Perú"])

with tab_tr_ch:
    st.markdown("**Chile – Tramos Transacción (BPS sobre monto)**")
    tramos_tr_ch = st.data_editor(
        tramos_transaccion["Chile"][["Tramo", "Min", "Max", "BPS"]],
        num_rows="fixed",
        use_container_width=True,
        key="tr_chile",
    )

with tab_tr_co:
    st.markdown("**Colombia – Tramos Transacción (BPS sobre monto)**")
    tramos_tr_co = st.data_editor(
        tramos_transaccion["Colombia"][["Tramo", "Min", "Max", "BPS"]],
        num_rows="fixed",
        use_container_width=True,
        key="tr_colombia",
    )

with tab_tr_pe:
    st.markdown("**Perú – Tramos Transacción (BPS sobre monto)**")
    tramos_tr_pe = st.data_editor(
        tramos_transaccion["Perú"][["Tramo", "Min", "Max", "BPS"]],
        num_rows="fixed",
        use_container_width=True,
        key="tr_peru",
    )


def actualizar_tramos_trans(df_edit: pd.DataFrame) -> pd.DataFrame:
    df = df_edit.copy()
    df["Tasa"] = df["BPS"] / 10_000
    return df


tramos_transaccion["Chile"] = actualizar_tramos_trans(tramos_tr_ch)
tramos_transaccion["Colombia"] = actualizar_tramos_trans(tramos_tr_co)
tramos_transaccion["Perú"] = actualizar_tramos_trans(tramos_tr_pe)
st.session_state["current_params"]["tramos_transaccion"] = tramos_transaccion


# ---------------------------------------------------------
# Configuración de alcance en Chile (DMA vs Total)
# ---------------------------------------------------------

st.markdown("---")
st.subheader("🇨🇱 Alcance de cobro en Chile")

scope_label = st.radio(
    "¿Sobre qué volumen se aplican las tarifas en Chile?",
    ["Solo DMA (como hoy)", "Todo el volumen (DMA + manual)"],
    index=0 if scope_chile == "DMA" else 1,
    horizontal=True,
)

scope_chile = "DMA" if "DMA" in scope_label.upper() else "TOTAL"
st.session_state["current_params"]["scope_chile"] = scope_chile

st.caption(
    "• **Solo DMA**: el ingreso proyectado de Chile solo se calcula sobre operaciones marcadas como DMA.\n"
    "• **Todo el volumen**: el ingreso proyectado usa todo el volumen negociado en Chile (DMA + no DMA)."
)


# ---------------------------------------------------------
# Gestión de escenarios (guardar / cargar)
# ---------------------------------------------------------

st.markdown("---")
st.subheader("📁 Gestión de escenarios")

col_es1, col_es2, col_es3 = st.columns([1.5, 1.2, 1.2])

with col_es1:
    escenario_nombre = st.text_input(
        "Nombre del escenario actual:",
        value=st.session_state.get("current_scenario_name", "Escenario 1"),
    )

with col_es2:
    if st.button("💾 Guardar / actualizar escenario"):
        if escenario_nombre:
            st.session_state["scenarios"][escenario_nombre] = {
                "tramos_acceso": {k: v.copy() for k, v in tramos_acceso.items()},
                "tramos_transaccion": {k: v.copy() for k, v in tramos_transaccion.items()},
                "scope_chile": scope_chile,
            }
            st.session_state["current_scenario_name"] = escenario_nombre
            st.success(f"Escenario '{escenario_nombre}' guardado/actualizado.")
        else:
            st.warning("Ponle un nombre al escenario antes de guardar.")

with col_es3:
    escenarios_disponibles = list(st.session_state["scenarios"].keys())
    escenario_sel = st.selectbox(
        "Escenario guardado:",
        options=["(ninguno)"] + escenarios_disponibles,
        index=0,
    )

c1, c2 = st.columns(2)
with c1:
    if st.button("📂 Cargar escenario seleccionado") and escenario_sel != "(ninguno)":
        esc = st.session_state["scenarios"][escenario_sel]
        st.session_state["current_params"] = {
            "tramos_acceso": {k: v.copy() for k, v in esc["tramos_acceso"].items()},
            "tramos_transaccion": {k: v.copy() for k, v in esc["tramos_transaccion"].items()},
            "scope_chile": esc.get("scope_chile", "DMA"),
        }
        st.session_state["current_scenario_name"] = escenario_sel
        st.experimental_rerun()

with c2:
    if st.button("🗑️ Borrar escenario seleccionado") and escenario_sel != "(ninguno)":
        st.session_state["scenarios"].pop(escenario_sel, None)
        st.success(f"Escenario '{escenario_sel}' eliminado.")
        st.experimental_rerun()

if st.session_state["scenarios"]:
    st.caption("Escenarios guardados en esta sesión: " + ", ".join(st.session_state["scenarios"].keys()))


# ---------------------------------------------------------
# Cálculo del escenario (Real vs Proyectado)
# ---------------------------------------------------------

with st.spinner("Calculando ingresos de Acceso y Transacción proyectados por corredor..."):
    df_proj_acceso = calcular_acceso_proyectado_por_tramos(df_bbdd, tramos_acceso, scope_chile)
    df_proj_trans = calcular_transaccion_proyectada_por_tramos(df_bbdd, tramos_transaccion, scope_chile)
    df_real = construir_real_desde_negociacion(df_neg, df_bbdd)
    df_brokers = unir_real_y_proyectado(df_real, df_proj_acceso, df_proj_trans)
    df_kpis = kpis_por_pais(df_brokers)


# ---------------------------------------------------------
# KPIs por país
# ---------------------------------------------------------

st.subheader("🏁 KPIs por País: Acceso + Transacción")

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
                "BPS_Acceso_Proyectado": [0.0], "BPS_Transaccion_Proyectada": [0.0], "BPS_Total_Proyectado": [0.0],
                "Ingreso_Real_Total": [0.0], "Ingreso_Proyectado_Total": [0.0],
            }),
        ], ignore_index=True)

df_kpis = df_kpis.set_index("Pais").loc[["Chile", "Colombia", "Perú"]].reset_index()

col1, col2, col3 = st.columns(3)

for col, pais in zip([col1, col2, col3], ["Chile", "Colombia", "Perú"]):
    row = df_kpis[df_kpis["Pais"] == pais].iloc[0]
    col.markdown(f"### {pais}")

    col.metric("Var % Acceso", f"{row['Var_%_Acceso']:,.1f}%")
    col.metric("Var % Transacción", f"{row['Var_%_Transaccion']:,.1f}%")
    col.metric("Var % TOTAL", f"{row['Var_%_Total']:,.1f}%")

    col.metric("BPS Acceso", f"{row['BPS_Acceso_Proyectado']:,.2f}")
    col.metric("BPS Transacción", f"{row['BPS_Transaccion_Proyectada']:,.2f}")
    col.metric("BPS TOTAL", f"{row['BPS_Total_Proyectado']:,.2f}")

st.markdown("---")


# ---------------------------------------------------------
# Tabla resumen de ingresos por país
# ---------------------------------------------------------

st.subheader("📊 Resumen de Ingresos por País")

df_resumen = df_kpis[[
    "Pais", "Monto_USD_Total",
    "Acceso_Real", "Acceso_Proyectado", "Var_%_Acceso",
    "Transaccion_Real", "Transaccion_Proyectada", "Var_%_Transaccion",
    "Ingreso_Real_Total", "Ingreso_Proyectado_Total", "Var_%_Total",
]].copy()

df_resumen.rename(columns={
    "Pais": "País",
    "Monto_USD_Total": "Monto Total (USD)",
    "Acceso_Real": "Acceso Real",
    "Acceso_Proyectado": "Acceso Proy.",
    "Var_%_Acceso": "Var% Acceso",
    "Transaccion_Real": "Trans. Real",
    "Transaccion_Proyectada": "Trans. Proy.",
    "Var_%_Transaccion": "Var% Trans.",
    "Ingreso_Real_Total": "Total Real",
    "Ingreso_Proyectado_Total": "Total Proy.",
    "Var_%_Total": "Var% Total",
}, inplace=True)

st.dataframe(
    df_resumen.style.format({
        "Monto Total (USD)": "${:,.0f}",
        "Acceso Real": "${:,.0f}",
        "Acceso Proy.": "${:,.0f}",
        "Var% Acceso": "{:,.1f}%",
        "Trans. Real": "${:,.0f}",
        "Trans. Proy.": "${:,.0f}",
        "Var% Trans.": "{:,.1f}%",
        "Total Real": "${:,.0f}",
        "Total Proy.": "${:,.0f}",
        "Var% Total": "{:,.1f}%",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")


# ---------------------------------------------------------
# Tabla detallada por broker
# ---------------------------------------------------------

st.subheader("📋 Detalle por Broker")

concepto_sel = st.radio(
    "Selecciona el concepto a mostrar:",
    ["Total", "Acceso", "Transaccion"],
    horizontal=True,
)

tab_b_ch, tab_b_co, tab_b_pe, tab_b_all = st.tabs(["Chile", "Colombia", "Perú", "Todos"])

with tab_b_ch:
    st.markdown("**Chile – Detalle por broker**")
    st.dataframe(preparar_tabla_brokers(df_brokers, "Chile", concepto_sel), use_container_width=True)

with tab_b_co:
    st.markdown("**Colombia – Detalle por broker**")
    st.dataframe(preparar_tabla_brokers(df_brokers, "Colombia", concepto_sel), use_container_width=True)

with tab_b_pe:
    st.markdown("**Perú – Detalle por broker**")
    st.dataframe(preparar_tabla_brokers(df_brokers, "Perú", concepto_sel), use_container_width=True)

with tab_b_all:
    st.markdown("**Todos los países – Detalle consolidado**")
    st.dataframe(preparar_tabla_brokers(df_brokers, None, concepto_sel), use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------

st.subheader("📈 Comparativo: Real vs Proyectado por País")

graf_tipo = st.selectbox("Concepto a graficar:", ["Acceso", "Transacción", "Total"])

if graf_tipo == "Acceso":
    df_chart = df_kpis[["Pais", "Acceso_Real", "Acceso_Proyectado"]].set_index("Pais")
    df_chart.columns = ["Real", "Proyectado"]
elif graf_tipo == "Transacción":
    df_chart = df_kpis[["Pais", "Transaccion_Real", "Transaccion_Proyectada"]].set_index("Pais")
    df_chart.columns = ["Real", "Proyectado"]
else:
    df_chart = df_kpis[["Pais", "Ingreso_Real_Total", "Ingreso_Proyectado_Total"]].set_index("Pais")
    df_chart.columns = ["Real", "Proyectado"]

st.bar_chart(df_chart)

st.markdown("---")


# ---------------------------------------------------------
# Gráfico de BPS
# ---------------------------------------------------------

st.subheader("📉 BPS Proyectado por País")

df_bps_chart = df_kpis[["Pais", "BPS_Acceso_Proyectado", "BPS_Transaccion_Proyectada"]].set_index("Pais")
df_bps_chart.columns = ["BPS Acceso", "BPS Transacción"]
st.bar_chart(df_bps_chart)

st.markdown("---")

st.caption(
    "**Lógica del simulador:**\n"
    "- **Acceso**: Para cada mes, se asigna una tarifa fija según el tramo de volumen del corredor. "
    "El ingreso anual es la suma de los 12 meses.\n"
    "- **Transacción**: Sistema de tramos **progresivos** (como impuestos). Cada tramo aplica solo "
    "al monto dentro de su rango.\n"
    "- **Chile DMA vs Total**: Puedes simular el esquema actual (solo DMA) o cobrar sobre todo el volumen "
    "simplemente cambiando la opción de alcance y/o guardando escenarios diferentes."
)
