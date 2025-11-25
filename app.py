import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional, Dict, List
import json

# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Tarifas RV - nuam",
    layout="wide",
)

st.title("📊 Simulador de Tarifas RV - nuam")
st.caption(
    "Simula tarifas de **Acceso**, **Transacción** y **DMA** para Chile, Colombia y Perú. "
    "Incluye análisis de escenarios, sensibilidad y comparativo multi-broker."
)


# ---------------------------------------------------------
# Funciones auxiliares de carga
# ---------------------------------------------------------

def load_bbdd_from_bytes(data: bytes, sheet_name: str = "A.3 BBDD Neg") -> pd.DataFrame:
    df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=6)
    return df


def load_param_sheet(excel_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(excel_bytes), sheet_name="1. Parametros", header=None)


def load_negociacion_sheet(excel_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(excel_bytes), sheet_name="3. Negociación", header=8)
    return df


# ---------------------------------------------------------
# Tramos de ACCESO desde 1. Parametros
# ---------------------------------------------------------

def get_tramos_acceso(df_params: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
                mx = 9999999999999.0
            if pd.isna(mn) or pd.isna(mx) or pd.isna(bps):
                continue
            data.append({
                "Tramo": f"Tramo {len(data) + 1}",
                "Min": float(mn),
                "Max": float(mx),
                "BPS": float(bps) * 10000,
                "Tasa": float(bps),
            })
        tramos_por_pais[pais] = pd.DataFrame(data)
    return tramos_por_pais


# ---------------------------------------------------------
# Tramos de DMA (solo Chile tiene volumen DMA)
# ---------------------------------------------------------

def get_tramos_dma(df_params: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    DMA tiene tramos progresivos con tasa variable.
    Solo Chile tiene volumen DMA significativo.
    Ubicación: Filas 140-142, columnas 27-30 para Chile.
    """
    tramos_por_pais: Dict[str, pd.DataFrame] = {}
    
    # Chile: cols 27 (Min), 28 (Max), 30 (Tasa variable)
    chile_data = []
    for r in [140, 141, 142]:
        mn = df_params.iat[r, 27]
        mx = df_params.iat[r, 28]
        tasa = df_params.iat[r, 30]
        
        if isinstance(mx, str):
            mx = 9999999999999.0
        if pd.isna(mn) or pd.isna(tasa):
            continue
        
        chile_data.append({
            "Tramo": f"Tramo {len(chile_data) + 1}",
            "Min": float(mn),
            "Max": float(mx) if not pd.isna(mx) else 9999999999999.0,
            "BPS": float(tasa) * 10000 if tasa != 0 else 0.0,
            "Tasa": float(tasa) if tasa != 0 else 0.0,
        })
    
    # Si no hay datos en el Excel, usar valores por defecto
    if not chile_data:
        chile_data = [
            {"Tramo": "Tramo 1", "Min": 0, "Max": 182361552.49, "BPS": 0.5, "Tasa": 0.00005},
            {"Tramo": "Tramo 2", "Min": 182361553.49, "Max": 364723104.98, "BPS": 0.3, "Tasa": 0.00003},
            {"Tramo": "Tramo 3", "Min": 364723105.98, "Max": 9999999999999.0, "BPS": 0.0, "Tasa": 0.0},
        ]
    
    tramos_por_pais["Chile"] = pd.DataFrame(chile_data)
    # Colombia y Perú no tienen DMA, pero creamos estructura vacía
    tramos_por_pais["Colombia"] = pd.DataFrame(columns=["Tramo", "Min", "Max", "BPS", "Tasa"])
    tramos_por_pais["Perú"] = pd.DataFrame(columns=["Tramo", "Min", "Max", "BPS", "Tasa"])
    
    return tramos_por_pais


# ---------------------------------------------------------
# Cálculo del ACCESO proyectado
# ---------------------------------------------------------

def calcular_acceso_proyectado_por_tramos(
    df_bbdd: pd.DataFrame, tramos_por_pais: Dict[str, pd.DataFrame], ajuste_pct: float = 0.0
) -> pd.DataFrame:
    df = df_bbdd.copy()
    df.rename(columns={"Pais": "Pais", "Corredor": "Broker", "Monto USD": "Monto_USD"}, inplace=True)
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
                return float(t["Fija_USD"]) * (1 + ajuste_pct / 100)
        return 0.0

    df["Acceso_Proyectado_mensual"] = df.apply(fee_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_USD", "Acceso_Proyectado_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(columns={"Monto_USD": "Monto_USD_Total", "Acceso_Proyectado_mensual": "Acceso_Proyectado"}, inplace=True)
    return agg


# ---------------------------------------------------------
# Cálculo de TRANSACCIÓN proyectada (progresivo)
# ---------------------------------------------------------

def calcular_transaccion_progresiva_mensual(monto: float, tramos_df: pd.DataFrame, ajuste_pct: float = 0.0) -> float:
    if tramos_df is None or tramos_df.empty:
        return 0.0
    
    tarifa_total = 0.0
    tramos_sorted = tramos_df.sort_values("Min").reset_index(drop=True)
    
    for _, tramo in tramos_sorted.iterrows():
        tramo_min = tramo["Min"]
        tramo_max = tramo["Max"]
        tasa = tramo["Tasa"] * (1 + ajuste_pct / 100)
        
        if monto <= tramo_min:
            continue
        
        monto_en_tramo = min(monto, tramo_max) - tramo_min
        monto_en_tramo = max(0, monto_en_tramo)
        tarifa_total += monto_en_tramo * tasa
    
    return tarifa_total


def calcular_transaccion_proyectada_por_tramos(
    df_bbdd: pd.DataFrame, tramos_por_pais: Dict[str, pd.DataFrame], ajuste_pct: float = 0.0
) -> pd.DataFrame:
    df = df_bbdd.copy()
    df.rename(columns={"Pais": "Pais", "Corredor": "Broker", "Monto USD": "Monto_USD"}, inplace=True)
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors="coerce").fillna(0.0)
    df["Pais"] = df["Pais"].astype(str).str.strip()

    def transaccion_for_row(row) -> float:
        pais = row["Pais"]
        monto = row["Monto_USD"]
        tramos_df = tramos_por_pais.get(pais)
        return calcular_transaccion_progresiva_mensual(monto, tramos_df, ajuste_pct)

    df["Transaccion_Proyectada_mensual"] = df.apply(transaccion_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_USD", "Transaccion_Proyectada_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(columns={"Monto_USD": "Monto_USD_Total", "Transaccion_Proyectada_mensual": "Transaccion_Proyectada"}, inplace=True)
    return agg


# ---------------------------------------------------------
# Cálculo de DMA proyectado (progresivo, solo Chile)
# ---------------------------------------------------------

def calcular_dma_proyectada_por_tramos(
    df_bbdd: pd.DataFrame, tramos_por_pais: Dict[str, pd.DataFrame], ajuste_pct: float = 0.0
) -> pd.DataFrame:
    df = df_bbdd.copy()
    df.rename(columns={"Pais": "Pais", "Corredor": "Broker", "Monto DMA USD": "Monto_DMA_USD"}, inplace=True)
    df["Monto_DMA_USD"] = pd.to_numeric(df["Monto_DMA_USD"], errors="coerce").fillna(0.0)
    df["Pais"] = df["Pais"].astype(str).str.strip()

    def dma_for_row(row) -> float:
        pais = row["Pais"]
        monto = row["Monto_DMA_USD"]
        tramos_df = tramos_por_pais.get(pais)
        if tramos_df is None or tramos_df.empty:
            return 0.0
        return calcular_transaccion_progresiva_mensual(monto, tramos_df, ajuste_pct)

    df["DMA_Proyectada_mensual"] = df.apply(dma_for_row, axis=1)

    agg = (
        df.groupby(["Pais", "Broker"])[["Monto_DMA_USD", "DMA_Proyectada_mensual"]]
        .sum()
        .reset_index()
    )
    agg.rename(columns={"Monto_DMA_USD": "Monto_DMA_Total", "DMA_Proyectada_mensual": "DMA_Proyectada"}, inplace=True)
    return agg


# ---------------------------------------------------------
# Construcción de "Real" desde 3. Negociación
# ---------------------------------------------------------

def construir_real_desde_negociacion(df_neg: pd.DataFrame, df_bbdd: pd.DataFrame) -> pd.DataFrame:
    broker_country = (
        df_bbdd.groupby("Corredor")["Pais"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    broker_country.rename(columns={"Corredor": "Broker"}, inplace=True)

    df_real = df_neg[["Corredor", "Real", "Real.1"]].copy()
    df_real.rename(columns={"Corredor": "Broker", "Real": "Acceso_Real", "Real.1": "Transaccion_Real"}, inplace=True)
    
    df_real["Acceso_Real"] = pd.to_numeric(df_real["Acceso_Real"], errors="coerce").fillna(0.0)
    df_real["Transaccion_Real"] = pd.to_numeric(df_real["Transaccion_Real"], errors="coerce").fillna(0.0)

    df_real = pd.merge(df_real, broker_country, on="Broker", how="left")
    df_real = df_real[~df_real["Pais"].isna()].copy()
    return df_real


# ---------------------------------------------------------
# Unión de Real y Proyectado
# ---------------------------------------------------------

def unir_real_y_proyectado(
    df_real: pd.DataFrame, 
    df_proj_acceso: pd.DataFrame, 
    df_proj_trans: pd.DataFrame,
    df_proj_dma: pd.DataFrame
) -> pd.DataFrame:
    
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
    
    df = pd.merge(
        df,
        df_proj_dma[["Pais", "Broker", "Monto_DMA_Total", "DMA_Proyectada"]],
        on=["Pais", "Broker"],
        how="outer",
    )

    for col in ["Acceso_Real", "Acceso_Proyectado", "Transaccion_Real", 
                "Transaccion_Proyectada", "Monto_USD_Total", "Monto_DMA_Total", "DMA_Proyectada"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Variaciones
    df["Var_%_Acceso"] = np.where(df["Acceso_Real"] != 0,
        (df["Acceso_Proyectado"] - df["Acceso_Real"]) / df["Acceso_Real"] * 100, 0.0)
    
    df["Var_%_Transaccion"] = np.where(df["Transaccion_Real"] != 0,
        (df["Transaccion_Proyectada"] - df["Transaccion_Real"]) / df["Transaccion_Real"] * 100, 0.0)
    
    # Totales
    df["Ingreso_Real_Total"] = df["Acceso_Real"] + df["Transaccion_Real"]
    df["Ingreso_Proyectado_Total"] = df["Acceso_Proyectado"] + df["Transaccion_Proyectada"] + df["DMA_Proyectada"]
    
    df["Var_%_Total"] = np.where(df["Ingreso_Real_Total"] != 0,
        (df["Ingreso_Proyectado_Total"] - df["Ingreso_Real_Total"]) / df["Ingreso_Real_Total"] * 100, 0.0)

    return df


def kpis_por_pais(df_brokers: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_brokers.groupby("Pais")[
            ["Monto_USD_Total", "Monto_DMA_Total", "Acceso_Real", "Acceso_Proyectado", 
             "Transaccion_Real", "Transaccion_Proyectada", "DMA_Proyectada"]
        ]
        .sum()
        .reset_index()
    )
    
    agg["Var_%_Acceso"] = np.where(agg["Acceso_Real"] != 0,
        (agg["Acceso_Proyectado"] - agg["Acceso_Real"]) / agg["Acceso_Real"] * 100, 0.0)
    
    agg["Var_%_Transaccion"] = np.where(agg["Transaccion_Real"] != 0,
        (agg["Transaccion_Proyectada"] - agg["Transaccion_Real"]) / agg["Transaccion_Real"] * 100, 0.0)
    
    agg["Ingreso_Real_Total"] = agg["Acceso_Real"] + agg["Transaccion_Real"]
    agg["Ingreso_Proyectado_Total"] = agg["Acceso_Proyectado"] + agg["Transaccion_Proyectada"] + agg["DMA_Proyectada"]
    
    agg["Var_%_Total"] = np.where(agg["Ingreso_Real_Total"] != 0,
        (agg["Ingreso_Proyectado_Total"] - agg["Ingreso_Real_Total"]) / agg["Ingreso_Real_Total"] * 100, 0.0)
    
    agg["BPS_Total_Proyectado"] = np.where(agg["Monto_USD_Total"] != 0,
        agg["Ingreso_Proyectado_Total"] / agg["Monto_USD_Total"] * 10000, 0.0)
    
    return agg


# ---------------------------------------------------------
# Datos mensuales para análisis temporal
# ---------------------------------------------------------

def obtener_datos_mensuales(df_bbdd: pd.DataFrame) -> pd.DataFrame:
    """Prepara datos mensuales para análisis de estacionalidad."""
    df = df_bbdd.copy()
    df["Monto USD"] = pd.to_numeric(df["Monto USD"], errors="coerce").fillna(0.0)
    df["Monto DMA USD"] = pd.to_numeric(df["Monto DMA USD"], errors="coerce").fillna(0.0)
    
    # Mapeo de meses a números
    meses_orden = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Setiembre': 9, 'Septiembre': 9, 'Octubre': 10, 
        'Noviembre': 11, 'Diciembre': 12
    }
    df["Mes_Num"] = df["Mes"].map(meses_orden)
    
    return df


def detectar_grupo_broker(nombre_broker) -> str:
    """Detecta el grupo corporativo del broker para análisis multi-país."""
    if pd.isna(nombre_broker):
        return "DESCONOCIDO"
    nombre_upper = str(nombre_broker).upper()
    
    if "CREDICORP" in nombre_upper:
        return "CREDICORP"
    elif "BTG PACTUAL" in nombre_upper:
        return "BTG PACTUAL"
    elif "LARRAIN VIAL" in nombre_upper:
        return "LARRAIN VIAL"
    elif "SCOTIA" in nombre_upper:
        return "SCOTIA"
    elif "ITAU" in nombre_upper:
        return "ITAU"
    elif "BBVA" in nombre_upper:
        return "BBVA"
    elif "SURA" in nombre_upper:
        return "SURA"
    else:
        return nombre_broker


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
# Inicializar session state para escenarios
# ---------------------------------------------------------

if "escenarios_guardados" not in st.session_state:
    st.session_state.escenarios_guardados = {}

if "ajuste_acceso" not in st.session_state:
    st.session_state.ajuste_acceso = 0.0

if "ajuste_transaccion" not in st.session_state:
    st.session_state.ajuste_transaccion = 0.0

if "ajuste_dma" not in st.session_state:
    st.session_state.ajuste_dma = 0.0


# ---------------------------------------------------------
# SIDEBAR: Configuración de tramos y sensibilidad
# ---------------------------------------------------------

st.sidebar.header("⚙️ Configuración")

# Cargar tramos base
tramos_acceso = get_tramos_acceso(df_params)
tramos_transaccion = get_tramos_transaccion(df_params)
tramos_dma = get_tramos_dma(df_params)

st.sidebar.subheader("📊 Análisis de Sensibilidad")
st.sidebar.markdown("*Ajusta las tarifas para ver el impacto*")

ajuste_acceso = st.sidebar.slider(
    "Ajuste Acceso (%)", 
    min_value=-50, max_value=100, value=0, step=5,
    help="Aumenta o reduce todas las tarifas de Acceso"
)

ajuste_transaccion = st.sidebar.slider(
    "Ajuste Transacción (%)", 
    min_value=-50, max_value=100, value=0, step=5,
    help="Aumenta o reduce todas las tarifas de Transacción"
)

ajuste_dma = st.sidebar.slider(
    "Ajuste DMA (%)", 
    min_value=-50, max_value=100, value=0, step=5,
    help="Aumenta o reduce todas las tarifas de DMA"
)

st.sidebar.markdown("---")

# Gestión de escenarios
st.sidebar.subheader("💾 Gestión de Escenarios")

nombre_escenario = st.sidebar.text_input("Nombre del escenario", value="Escenario 1")

if st.sidebar.button("💾 Guardar escenario actual"):
    st.session_state.escenarios_guardados[nombre_escenario] = {
        "ajuste_acceso": ajuste_acceso,
        "ajuste_transaccion": ajuste_transaccion,
        "ajuste_dma": ajuste_dma,
    }
    st.sidebar.success(f"✅ Escenario '{nombre_escenario}' guardado")

if st.session_state.escenarios_guardados:
    st.sidebar.markdown("**Escenarios guardados:**")
    for nombre, config in st.session_state.escenarios_guardados.items():
        st.sidebar.markdown(f"- {nombre}: Acc={config['ajuste_acceso']}%, Trans={config['ajuste_transaccion']}%, DMA={config['ajuste_dma']}%")


# ---------------------------------------------------------
# Cálculos principales
# ---------------------------------------------------------

with st.spinner("Calculando ingresos proyectados..."):
    df_proj_acceso = calcular_acceso_proyectado_por_tramos(df_bbdd, tramos_acceso, ajuste_acceso)
    df_proj_trans = calcular_transaccion_proyectada_por_tramos(df_bbdd, tramos_transaccion, ajuste_transaccion)
    df_proj_dma = calcular_dma_proyectada_por_tramos(df_bbdd, tramos_dma, ajuste_dma)
    df_real = construir_real_desde_negociacion(df_neg, df_bbdd)
    df_brokers = unir_real_y_proyectado(df_real, df_proj_acceso, df_proj_trans, df_proj_dma)
    df_kpis = kpis_por_pais(df_brokers)


# ---------------------------------------------------------
# TABS principales
# ---------------------------------------------------------

tab_kpis, tab_brokers, tab_mensual, tab_escenarios, tab_tramos = st.tabs([
    "📊 KPIs por País", 
    "👥 Detalle Brokers", 
    "📅 Análisis Mensual Multi-Broker",
    "🔄 Comparar Escenarios",
    "⚙️ Configurar Tramos"
])


# ---------------------------------------------------------
# TAB 1: KPIs por país
# ---------------------------------------------------------

with tab_kpis:
    st.subheader("🏁 KPIs por País: Acceso + Transacción + DMA")
    
    # Mostrar ajustes actuales
    if ajuste_acceso != 0 or ajuste_transaccion != 0 or ajuste_dma != 0:
        st.info(f"📊 **Ajustes aplicados:** Acceso {ajuste_acceso:+d}%, Transacción {ajuste_transaccion:+d}%, DMA {ajuste_dma:+d}%")
    
    # Asegurar los 3 países
    for p in ["Chile", "Colombia", "Perú"]:
        if p not in df_kpis["Pais"].values:
            df_kpis = pd.concat([df_kpis, pd.DataFrame({
                "Pais": [p], "Monto_USD_Total": [0.0], "Monto_DMA_Total": [0.0],
                "Acceso_Real": [0.0], "Acceso_Proyectado": [0.0],
                "Transaccion_Real": [0.0], "Transaccion_Proyectada": [0.0],
                "DMA_Proyectada": [0.0], "Var_%_Acceso": [0.0], "Var_%_Transaccion": [0.0],
                "Var_%_Total": [0.0], "Ingreso_Real_Total": [0.0], 
                "Ingreso_Proyectado_Total": [0.0], "BPS_Total_Proyectado": [0.0],
            })], ignore_index=True)

    df_kpis = df_kpis.set_index("Pais").loc[["Chile", "Colombia", "Perú"]].reset_index()

    # Métricas por país
    col1, col2, col3 = st.columns(3)

    for col, pais in zip([col1, col2, col3], ["Chile", "Colombia", "Perú"]):
        row = df_kpis[df_kpis["Pais"] == pais].iloc[0]
        col.markdown(f"### {pais}")
        
        delta_color = "normal" if row['Var_%_Total'] >= 0 else "inverse"
        col.metric("Ingreso Proyectado", f"${row['Ingreso_Proyectado_Total']:,.0f}", 
                   f"{row['Var_%_Total']:+.1f}% vs Real")
        col.metric("BPS Total", f"{row['BPS_Total_Proyectado']:.2f}")
        
        # Desglose
        col.markdown("**Desglose:**")
        col.markdown(f"- Acceso: ${row['Acceso_Proyectado']:,.0f} ({row['Var_%_Acceso']:+.1f}%)")
        col.markdown(f"- Transacción: ${row['Transaccion_Proyectada']:,.0f} ({row['Var_%_Transaccion']:+.1f}%)")
        if row['DMA_Proyectada'] > 0:
            col.markdown(f"- DMA: ${row['DMA_Proyectada']:,.0f}")

    st.markdown("---")
    
    # Tabla resumen
    st.subheader("📊 Resumen de Ingresos por País")
    
    df_resumen = df_kpis[["Pais", "Monto_USD_Total", "Monto_DMA_Total",
                          "Acceso_Proyectado", "Transaccion_Proyectada", "DMA_Proyectada",
                          "Ingreso_Proyectado_Total", "Ingreso_Real_Total", "Var_%_Total"]].copy()
    
    df_resumen.columns = ["País", "Monto Total", "Monto DMA", "Acceso", "Transacción", "DMA", 
                          "Total Proyectado", "Total Real", "Var%"]
    
    st.dataframe(
        df_resumen.style.format({
            "Monto Total": "${:,.0f}",
            "Monto DMA": "${:,.0f}",
            "Acceso": "${:,.0f}",
            "Transacción": "${:,.0f}",
            "DMA": "${:,.0f}",
            "Total Proyectado": "${:,.0f}",
            "Total Real": "${:,.0f}",
            "Var%": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    # Gráfico comparativo
    st.subheader("📈 Real vs Proyectado por País")
    df_chart = df_kpis[["Pais", "Ingreso_Real_Total", "Ingreso_Proyectado_Total"]].set_index("Pais")
    df_chart.columns = ["Real", "Proyectado"]
    st.bar_chart(df_chart)


# ---------------------------------------------------------
# TAB 2: Detalle Brokers
# ---------------------------------------------------------

with tab_brokers:
    st.subheader("👥 Detalle por Broker")
    
    pais_sel = st.selectbox("Filtrar por país:", ["Todos", "Chile", "Colombia", "Perú"])
    
    df_view = df_brokers.copy()
    if pais_sel != "Todos":
        df_view = df_view[df_view["Pais"] == pais_sel]
    
    df_tabla = df_view[[
        "Pais", "Broker", "Monto_USD_Total", "Monto_DMA_Total",
        "Acceso_Proyectado", "Transaccion_Proyectada", "DMA_Proyectada",
        "Ingreso_Proyectado_Total", "Ingreso_Real_Total", "Var_%_Total"
    ]].copy()
    
    df_tabla.columns = ["País", "Broker", "Monto USD", "Monto DMA", "Acceso", "Trans.", "DMA", 
                        "Total Proy.", "Total Real", "Var%"]
    
    df_tabla = df_tabla.sort_values("Monto USD", ascending=False)
    
    st.dataframe(
        df_tabla.style.format({
            "Monto USD": "${:,.0f}",
            "Monto DMA": "${:,.0f}",
            "Acceso": "${:,.0f}",
            "Trans.": "${:,.0f}",
            "DMA": "${:,.0f}",
            "Total Proy.": "${:,.0f}",
            "Total Real": "${:,.0f}",
            "Var%": "{:+.1f}%",
        }).applymap(
            lambda x: 'background-color: #ffcccc' if isinstance(x, str) and '-' in x and float(x.replace('%','').replace('+','')) < -20 
            else ('background-color: #ccffcc' if isinstance(x, str) and '+' in x and float(x.replace('%','').replace('+','')) > 20 else ''),
            subset=["Var%"]
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# TAB 3: Análisis Mensual Multi-Broker
# ---------------------------------------------------------

with tab_mensual:
    st.subheader("📅 Análisis Mensual por Broker (Multi-País)")
    
    st.markdown("""
    Selecciona múltiples brokers de diferentes países para comparar su comportamiento.
    Ideal para analizar grupos corporativos como **Credicorp**, **BTG Pactual**, **Larrain Vial**, etc.
    """)
    
    # Preparar datos mensuales
    df_mensual = obtener_datos_mensuales(df_bbdd)
    
    # Agregar columna de grupo
    df_mensual["Grupo"] = df_mensual["Corredor"].apply(detectar_grupo_broker)
    
    # Lista de brokers disponibles con su país
    brokers_disponibles = df_mensual.groupby(["Corredor", "Pais"]).size().reset_index()[["Corredor", "Pais"]]
    brokers_disponibles["Broker_Display"] = brokers_disponibles["Corredor"] + " (" + brokers_disponibles["Pais"] + ")"
    
    # Detectar grupos multi-país
    grupos_multi = df_mensual.groupby("Grupo")["Pais"].nunique()
    grupos_multi = grupos_multi[grupos_multi > 1].index.tolist()
    
    st.markdown("**🌎 Grupos con presencia multi-país detectados:**")
    st.markdown(", ".join([f"`{g}`" for g in grupos_multi]))
    
    # Selector de brokers
    col_sel1, col_sel2 = st.columns([1, 1])
    
    with col_sel1:
        # Opción 1: Seleccionar por grupo
        grupo_seleccionado = st.selectbox(
            "Seleccionar por grupo corporativo:",
            ["(Selección manual)"] + grupos_multi
        )
    
    with col_sel2:
        # Opción 2: Selección manual
        if grupo_seleccionado == "(Selección manual)":
            brokers_seleccionados = st.multiselect(
                "Seleccionar brokers:",
                options=brokers_disponibles["Broker_Display"].tolist(),
                default=[]
            )
        else:
            # Auto-seleccionar brokers del grupo
            brokers_del_grupo = df_mensual[df_mensual["Grupo"] == grupo_seleccionado]["Corredor"].unique()
            brokers_seleccionados = [f"{b} ({df_mensual[df_mensual['Corredor']==b]['Pais'].iloc[0]})" 
                                     for b in brokers_del_grupo]
            st.info(f"Auto-seleccionados {len(brokers_seleccionados)} brokers del grupo {grupo_seleccionado}")
    
    if brokers_seleccionados:
        # Extraer nombres de brokers sin el país
        brokers_filtro = [b.rsplit(" (", 1)[0] for b in brokers_seleccionados]
        
        # Filtrar datos
        df_filtrado = df_mensual[df_mensual["Corredor"].isin(brokers_filtro)]
        
        # Agregar por mes y país
        df_mes_pais = df_filtrado.groupby(["Mes_Num", "Mes", "Pais"]).agg({
            "Monto USD": "sum",
            "Monto DMA USD": "sum",
            "Corredor": "nunique"
        }).reset_index()
        df_mes_pais.columns = ["Mes_Num", "Mes", "País", "Monto USD", "Monto DMA", "N_Brokers"]
        df_mes_pais = df_mes_pais.sort_values("Mes_Num")
        
        # Total consolidado
        df_mes_total = df_filtrado.groupby(["Mes_Num", "Mes"]).agg({
            "Monto USD": "sum",
            "Monto DMA USD": "sum",
        }).reset_index()
        df_mes_total = df_mes_total.sort_values("Mes_Num")
        
        # Mostrar KPIs del grupo
        st.markdown("### 📊 Resumen del Grupo Seleccionado")
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        total_monto = df_filtrado["Monto USD"].sum()
        total_dma = df_filtrado["Monto DMA USD"].sum()
        
        col_k1.metric("Monto Total USD", f"${total_monto/1e9:.2f}B")
        col_k2.metric("Monto DMA USD", f"${total_dma/1e9:.2f}B")
        col_k3.metric("% DMA", f"{total_dma/total_monto*100:.1f}%" if total_monto > 0 else "0%")
        col_k4.metric("Brokers", len(brokers_filtro))
        
        # Desglose por país
        st.markdown("### 🌎 Desglose por País")
        
        df_por_pais = df_filtrado.groupby("Pais").agg({
            "Monto USD": "sum",
            "Monto DMA USD": "sum",
            "Corredor": lambda x: x.unique().tolist()
        }).reset_index()
        
        for _, row in df_por_pais.iterrows():
            pais = row["Pais"]
            monto = row["Monto USD"]
            monto_dma = row["Monto DMA USD"]
            brokers_pais = row["Corredor"]
            
            # Calcular ingreso proyectado para este broker en este país
            df_broker_pais = df_brokers[(df_brokers["Pais"] == pais) & (df_brokers["Broker"].isin(brokers_pais))]
            
            ingreso_proy = df_broker_pais["Ingreso_Proyectado_Total"].sum()
            ingreso_real = df_broker_pais["Ingreso_Real_Total"].sum()
            var_pct = ((ingreso_proy - ingreso_real) / ingreso_real * 100) if ingreso_real > 0 else 0
            
            # Determinar color según variación
            if var_pct > 10:
                emoji = "🟢"
                color = "green"
            elif var_pct < -10:
                emoji = "🔴"
                color = "red"
            else:
                emoji = "🟡"
                color = "orange"
            
            st.markdown(f"""
            **{emoji} {pais}:** Monto ${monto/1e6:.1f}M | Ingreso Proy. ${ingreso_proy:,.0f} | 
            Var: <span style='color:{color}'>{var_pct:+.1f}%</span>
            """, unsafe_allow_html=True)
        
        # Gráfico de evolución mensual
        st.markdown("### 📈 Evolución Mensual por País")
        
        # Pivot para gráfico
        df_pivot = df_mes_pais.pivot(index="Mes", columns="País", values="Monto USD")
        # Reordenar meses
        meses_orden_lista = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                            'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df_pivot = df_pivot.reindex([m for m in meses_orden_lista if m in df_pivot.index])
        
        st.line_chart(df_pivot)
        
        # Tabla detallada
        st.markdown("### 📋 Detalle Mensual")
        
        st.dataframe(
            df_mes_pais[["Mes", "País", "Monto USD", "Monto DMA"]].style.format({
                "Monto USD": "${:,.0f}",
                "Monto DMA": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("👆 Selecciona uno o más brokers para ver el análisis mensual")


# ---------------------------------------------------------
# TAB 4: Comparar Escenarios
# ---------------------------------------------------------

with tab_escenarios:
    st.subheader("🔄 Comparación de Escenarios")
    
    if len(st.session_state.escenarios_guardados) < 2:
        st.warning("⚠️ Necesitas guardar al menos 2 escenarios en el panel lateral para comparar.")
        st.markdown("""
        **Cómo crear escenarios:**
        1. Ajusta los sliders de sensibilidad en el panel lateral
        2. Dale un nombre al escenario
        3. Haz clic en "Guardar escenario actual"
        4. Repite para crear más escenarios
        """)
    else:
        # Selector de escenarios a comparar
        escenarios_disponibles = list(st.session_state.escenarios_guardados.keys())
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            escenario_a = st.selectbox("Escenario A:", escenarios_disponibles, index=0)
        with col_e2:
            escenario_b = st.selectbox("Escenario B:", escenarios_disponibles, 
                                       index=min(1, len(escenarios_disponibles)-1))
        
        if st.button("🔄 Comparar Escenarios"):
            config_a = st.session_state.escenarios_guardados[escenario_a]
            config_b = st.session_state.escenarios_guardados[escenario_b]
            
            # Calcular para escenario A
            df_acc_a = calcular_acceso_proyectado_por_tramos(df_bbdd, tramos_acceso, config_a["ajuste_acceso"])
            df_trans_a = calcular_transaccion_proyectada_por_tramos(df_bbdd, tramos_transaccion, config_a["ajuste_transaccion"])
            df_dma_a = calcular_dma_proyectada_por_tramos(df_bbdd, tramos_dma, config_a["ajuste_dma"])
            df_brokers_a = unir_real_y_proyectado(df_real, df_acc_a, df_trans_a, df_dma_a)
            kpis_a = kpis_por_pais(df_brokers_a)
            
            # Calcular para escenario B
            df_acc_b = calcular_acceso_proyectado_por_tramos(df_bbdd, tramos_acceso, config_b["ajuste_acceso"])
            df_trans_b = calcular_transaccion_proyectada_por_tramos(df_bbdd, tramos_transaccion, config_b["ajuste_transaccion"])
            df_dma_b = calcular_dma_proyectada_por_tramos(df_bbdd, tramos_dma, config_b["ajuste_dma"])
            df_brokers_b = unir_real_y_proyectado(df_real, df_acc_b, df_trans_b, df_dma_b)
            kpis_b = kpis_por_pais(df_brokers_b)
            
            # Mostrar comparación
            st.markdown("### 📊 Comparación de Resultados")
            
            # Configuraciones
            st.markdown(f"""
            | Parámetro | **{escenario_a}** | **{escenario_b}** |
            |-----------|-------------------|-------------------|
            | Ajuste Acceso | {config_a['ajuste_acceso']:+d}% | {config_b['ajuste_acceso']:+d}% |
            | Ajuste Transacción | {config_a['ajuste_transaccion']:+d}% | {config_b['ajuste_transaccion']:+d}% |
            | Ajuste DMA | {config_a['ajuste_dma']:+d}% | {config_b['ajuste_dma']:+d}% |
            """)
            
            # Resultados por país
            st.markdown("### 💰 Ingresos Proyectados por País")
            
            for pais in ["Chile", "Colombia", "Perú"]:
                kpi_a = kpis_a[kpis_a["Pais"] == pais].iloc[0] if pais in kpis_a["Pais"].values else None
                kpi_b = kpis_b[kpis_b["Pais"] == pais].iloc[0] if pais in kpis_b["Pais"].values else None
                
                if kpi_a is not None and kpi_b is not None:
                    ing_a = kpi_a["Ingreso_Proyectado_Total"]
                    ing_b = kpi_b["Ingreso_Proyectado_Total"]
                    diff = ing_b - ing_a
                    diff_pct = (diff / ing_a * 100) if ing_a > 0 else 0
                    
                    emoji = "🟢" if diff > 0 else ("🔴" if diff < 0 else "🟡")
                    
                    st.markdown(f"""
                    **{pais}:**
                    - {escenario_a}: ${ing_a:,.0f}
                    - {escenario_b}: ${ing_b:,.0f}
                    - Diferencia: {emoji} ${diff:+,.0f} ({diff_pct:+.1f}%)
                    """)
            
            # Totales
            total_a = kpis_a["Ingreso_Proyectado_Total"].sum()
            total_b = kpis_b["Ingreso_Proyectado_Total"].sum()
            diff_total = total_b - total_a
            
            st.markdown("---")
            st.markdown(f"""
            ### 🏆 TOTAL CONSOLIDADO
            - **{escenario_a}:** ${total_a:,.0f}
            - **{escenario_b}:** ${total_b:,.0f}
            - **Diferencia:** ${diff_total:+,.0f} ({diff_total/total_a*100:+.1f}%)
            """)
            
            # Gráfico comparativo
            df_comp = pd.DataFrame({
                "País": ["Chile", "Colombia", "Perú"],
                escenario_a: [kpis_a[kpis_a["Pais"]==p]["Ingreso_Proyectado_Total"].iloc[0] if p in kpis_a["Pais"].values else 0 for p in ["Chile", "Colombia", "Perú"]],
                escenario_b: [kpis_b[kpis_b["Pais"]==p]["Ingreso_Proyectado_Total"].iloc[0] if p in kpis_b["Pais"].values else 0 for p in ["Chile", "Colombia", "Perú"]],
            }).set_index("País")
            
            st.bar_chart(df_comp)


# ---------------------------------------------------------
# TAB 5: Configurar Tramos
# ---------------------------------------------------------

with tab_tramos:
    st.subheader("⚙️ Configuración de Tramos Tarifarios")
    
    st.markdown("""
    Aquí puedes ver y modificar los tramos base. Los cambios en los sliders de sensibilidad 
    se aplican como porcentaje sobre estos valores base.
    """)
    
    concepto_editar = st.selectbox("Selecciona el concepto a editar:", ["Acceso", "Transacción", "DMA"])
    
    if concepto_editar == "Acceso":
        st.markdown("### 🔐 Tramos de Acceso (Tarifa Fija Mensual)")
        
        tab_acc_ch, tab_acc_co, tab_acc_pe = st.tabs(["Chile", "Colombia", "Perú"])
        
        with tab_acc_ch:
            st.data_editor(tramos_acceso["Chile"], num_rows="fixed", use_container_width=True, key="ed_acc_ch")
        with tab_acc_co:
            st.data_editor(tramos_acceso["Colombia"], num_rows="fixed", use_container_width=True, key="ed_acc_co")
        with tab_acc_pe:
            st.data_editor(tramos_acceso["Perú"], num_rows="fixed", use_container_width=True, key="ed_acc_pe")
    
    elif concepto_editar == "Transacción":
        st.markdown("### 💱 Tramos de Transacción (BPS Progresivos)")
        st.info("Sistema progresivo: cada tramo aplica solo al monto dentro de su rango")
        
        tab_tr_ch, tab_tr_co, tab_tr_pe = st.tabs(["Chile", "Colombia", "Perú"])
        
        with tab_tr_ch:
            st.data_editor(tramos_transaccion["Chile"][["Tramo", "Min", "Max", "BPS"]], 
                          num_rows="fixed", use_container_width=True, key="ed_tr_ch")
        with tab_tr_co:
            st.data_editor(tramos_transaccion["Colombia"][["Tramo", "Min", "Max", "BPS"]], 
                          num_rows="fixed", use_container_width=True, key="ed_tr_co")
        with tab_tr_pe:
            st.data_editor(tramos_transaccion["Perú"][["Tramo", "Min", "Max", "BPS"]], 
                          num_rows="fixed", use_container_width=True, key="ed_tr_pe")
    
    else:  # DMA
        st.markdown("### 📡 Tramos de DMA (Solo Chile)")
        st.info("DMA solo tiene volumen significativo en Chile (67% del volumen total de Chile)")
        
        st.data_editor(tramos_dma["Chile"][["Tramo", "Min", "Max", "BPS"]], 
                      num_rows="fixed", use_container_width=True, key="ed_dma_ch")
        
        # Mostrar estadística de DMA
        monto_dma_chile = df_bbdd[df_bbdd["Pais"] == "Chile"]["Monto DMA USD"].sum()
        monto_total_chile = df_bbdd[df_bbdd["Pais"] == "Chile"]["Monto USD"].sum()
        
        st.markdown(f"""
        **Estadísticas DMA Chile:**
        - Monto DMA: ${monto_dma_chile/1e9:.2f}B USD
        - Monto Total: ${monto_total_chile/1e9:.2f}B USD
        - % DMA: {monto_dma_chile/monto_total_chile*100:.1f}%
        """)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")
st.caption(
    "**Simulador de Tarifas RV nuam v2.0** | "
    "Incluye Acceso, Transacción y DMA | "
    "Análisis de sensibilidad y comparación de escenarios"
)
