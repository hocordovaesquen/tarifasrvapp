import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador Tarifas Acceso RV - nuam",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Simulador de Tarifas de Acceso RV - nuam")
st.markdown("""
**Instrucciones:**
1. Sube tu archivo Excel "Modelamiento Estructura Tarifaria".
2. El sistema tomará los **Ingresos Reales** de la hoja `3. Negociación` (Tu fuente de verdad).
3. Recalculará el escenario propuesto usando los volúmenes de la `BBDD` y tus parámetros.
""")

# ---------------------------------------------------------
# Funciones de Carga y Limpieza (Backend)
# ---------------------------------------------------------

def normalize_text(text):
    """Ayuda a cruzar nombres de brokers entre hojas (quita espacios, mayusculas, S.A., etc)"""
    if pd.isna(text): return ""
    text = str(text).upper().strip()
    # Quitar sufijos comunes para mejorar el match
    text = re.sub(r'\s+S\.?A\.?.*', '', text) 
    text = re.sub(r'\s+CORREDORES.*', '', text)
    text = re.sub(r'\s+S\.?A\.?B\.?.*', '', text)
    return text

def find_header_row(df, keywords, max_rows=20):
    """Busca en qué fila empiezan los títulos"""
    for i, row in df.head(max_rows).iterrows():
        row_str = row.astype(str).str.lower().values.tolist()
        # Si encuentra al menos 2 palabras clave en la misma fila
        match_count = sum(1 for k in keywords if any(k in s for s in row_str))
        if match_count >= 1:
            return i
    return 7 # Fallback

def load_data(excel_bytes):
    """
    Carga DOS fuentes de información:
    1. BBDD: Para obtener volúmenes y calcular lo NUEVO.
    2. Hoja '3. Negociación': Para obtener lo REAL (Actual).
    """
    xls = pd.ExcelFile(BytesIO(excel_bytes))
    
    # --- 1. CARGAR BBDD (Para Volúmenes y Proyección) ---
    sheet_bbdd = "A.3 BBDD Neg"
    if sheet_bbdd not in xls.sheet_names:
        st.error(f"No se encontró la hoja '{sheet_bbdd}'")
        return pd.DataFrame(), pd.DataFrame()

    # Detectar header dinámicamente
    raw_bbdd = pd.read_excel(xls, sheet_name=sheet_bbdd, header=None, nrows=20)
    idx_bbdd = find_header_row(raw_bbdd, ["pais", "corredor", "cliente"])
    df_bbdd = pd.read_excel(xls, sheet_name=sheet_bbdd, header=idx_bbdd)
    
    # Limpieza columnas BBDD
    df_bbdd.columns = [str(c).strip() for c in df_bbdd.columns]
    
    # --- 2. CARGAR REALES (Hoja 3. Negociación) ---
    sheet_real = "3. Negociación"
    df_real_clean = pd.DataFrame()
    
    if sheet_real in xls.sheet_names:
        # Leemos la hoja buscando encabezados como "Corredor" y "Acceso"
        raw_real = pd.read_excel(xls, sheet_name=sheet_real, header=None, nrows=30)
        idx_real = find_header_row(raw_real, ["corredor", "broker", "acceso", "total"])
        
        df_real = pd.read_excel(xls, sheet_name=sheet_real, header=idx_real)
        df_real.columns = [str(c).strip() for c in df_real.columns]
        
        # Identificar columnas clave en la hoja Real
        col_broker_r = next((c for c in df_real.columns if "corredor" in c.lower() or "broker" in c.lower()), None)
        # Buscamos columnas que digan "Acceso" y "nuam" o "actual" o simplemente "Acceso"
        col_acceso_r = next((c for c in df_real.columns if "acceso" in c.lower() and ("nuam" in c.lower() or "real" in c.lower() or "actual" in c.lower())), None)
        
        # Si no encuentra "Acceso nuam", busca solo "Acceso" (Cuidado con columnas de otros países)
        if not col_acceso_r:
             col_acceso_r = next((c for c in df_real.columns if "acceso" in c.lower()), None)

        if col_broker_r and col_acceso_r:
            df_real_clean = df_real[[col_broker_r, col_acceso_r]].copy()
            df_real_clean.columns = ["Broker_Join", "Acceso_Real_Sheet"]
            df_real_clean["Broker_Key"] = df_real_clean["Broker_Join"].apply(normalize_text)
            # Limpiar valores numéricos
            df_real_clean["Acceso_Real_Sheet"] = pd.to_numeric(df_real_clean["Acceso_Real_Sheet"], errors='coerce').fillna(0)
    
    return df_bbdd, df_real_clean

# ---------------------------------------------------------
# Lógica de Cálculo (Híbrida)
# ---------------------------------------------------------

def calcular_escenario(df_base: pd.DataFrame, df_reales: pd.DataFrame, params: dict):
    
    df = df_base.copy()
    
    # 1. Normalizar columnas de la BBDD
    col_map = {c.lower(): c for c in df.columns}
    col_pais = next((col_map[c] for c in col_map if "pais" in c or "país" in c), "Pais")
    col_broker = next((col_map[c] for c in col_map if "corredor" in c or "broker" in c or "cliente" in c), "Corredor")
    col_monto = next((col_map[c] for c in col_map if "monto usd" in c or "volumen usd" in c), "Monto USD")
    col_mes = next((col_map[c] for c in col_map if "mes" in c or "month" in c), "Mes")

    df.rename(columns={col_pais:"Pais", col_broker:"Broker", col_monto:"Monto_USD", col_mes:"Mes"}, inplace=True)
    
    # Limpieza
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors='coerce').fillna(0)
    df["Pais"] = df["Pais"].astype(str).str.upper().str.strip().replace({"PERU":"Perú", "PERÚ":"Perú", "CHILE":"Chile", "COLOMBIA":"Colombia"})
    df = df[df["Pais"].isin(["Chile", "Colombia", "Perú"])]

    # -----------------------------------------------------------------
    # PARTE A: CALCULAR PROYECCIÓN (Usando BBDD y Tramos Mensuales)
    # -----------------------------------------------------------------
    
    # Agrupar por mes para aplicar tramos
    df_mensual = df.groupby(["Pais", "Broker", "Mes"])["Monto_USD"].sum().reset_index()

    def get_tarifa(row):
        cfg = params.get(row["Pais"])
        if not cfg: return 0.0
        vol = row["Monto_USD"]
        if vol < cfg["limite_bajo"]: return cfg["tarifa_bajo"]
        elif vol < cfg["limite_alto"]: return cfg["tarifa_medio"]
        else: return cfg["tarifa_alto"]

    df_mensual["Acceso_Proyectado"] = df_mensual.apply(get_tarifa, axis=1)

    # Consolidar Año
    df_final = df_mensual.groupby(["Pais", "Broker"]).agg({
        "Monto_USD": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()

    # -----------------------------------------------------------------
    # PARTE B: INYECTAR DATOS REALES (Desde hoja 3. Negociación)
    # -----------------------------------------------------------------
    
    if not df_reales.empty:
        # Crear llave de cruce
        df_final["Broker_Key"] = df_final["Broker"].apply(normalize_text)
        
        # Merge con la hoja real
        df_merged = pd.merge(df_final, df_reales[["Broker_Key", "Acceso_Real_Sheet"]], on="Broker_Key", how="left")
        
        # Si cruzó, usamos el dato de la hoja Real. Si no, usamos 0 (o lo que quieras definir)
        df_merged["Acceso_Actual"] = df_merged["Acceso_Real_Sheet"].fillna(0)
        
        # Limpieza aux
        df_final = df_merged.drop(columns=["Broker_Key", "Acceso_Real_Sheet", "Broker_Join"], errors='ignore')
    else:
        # Si falló la carga de hoja real, ponemos 0
        df_final["Acceso_Actual"] = 0.0

    # -----------------------------------------------------------------
    # PARTE C: KPIs
    # -----------------------------------------------------------------
    
    df_final["Var_Abs"] = df_final["Acceso_Proyectado"] - df_final["Acceso_Actual"]
    df_final["Var_%"] = np.where(df_final["Acceso_Actual"] > 1, (df_final["Var_Abs"]/df_final["Acceso_Actual"])*100, 0.0)

    # Resumen Pais
    agg_pais = df_final.groupby("Pais").agg({
        "Monto_USD":"sum", "Acceso_Actual":"sum", "Acceso_Proyectado":"sum"
    }).reset_index()
    agg_pais["Var_%"] = ((agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]) / agg_pais["Acceso_Actual"]) * 100

    return df_final, agg_pais

# ---------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------

uploaded_file = st.file_uploader("📥 Cargar Excel Completo", type=["xlsx"])

if uploaded_file:
    df_bbdd_raw, df_real_raw = load_data(uploaded_file.getvalue())

    if df_bbdd_raw.empty:
        st.error("Error cargando BBDD.")
        st.stop()
    
    if df_real_raw.empty:
        st.warning("⚠️ No se pudo leer automáticamente la columna de 'Acceso' en la hoja '3. Negociación'. Los valores Actuales saldrán en 0. Revisa que la columna tenga un título reconocible.")
    else:
        st.success(f"✅ Datos Reales cargados desde '3. Negociación' ({len(df_real_raw)} registros).")

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("⚙️ Configuración de Tramos")
    params = {}
    defaults = {
        "Chile":    {"L1":5.0, "L2":15.0, "T1":500.0, "T2":1500.0, "T3":2500.0},
        "Colombia": {"L1":5.0, "L2":15.0, "T1":400.0, "T2":1200.0, "T3":2000.0},
        "Perú":     {"L1":5.0, "L2":15.0, "T1":350.0, "T2":1500.0, "T3":2000.0}
    }
    
    for pais in ["Chile", "Colombia", "Perú"]:
        with st.sidebar.expander(f"{pais}", expanded=False):
            l1 = st.number_input(f"Límite 1 (MM) {pais}", value=defaults[pais]["L1"])
            l2 = st.number_input(f"Límite 2 (MM) {pais}", value=defaults[pais]["L2"])
            t1 = st.number_input(f"Tarifa Baja {pais}", value=defaults[pais]["T1"])
            t2 = st.number_input(f"Tarifa Media {pais}", value=defaults[pais]["T2"])
            t3 = st.number_input(f"Tarifa Alta {pais}", value=defaults[pais]["T3"])
            params[pais] = {"limite_bajo": l1*1e6, "limite_alto": l2*1e6, "tarifa_bajo": t1, "tarifa_medio": t2, "tarifa_alto": t3}

    # --- CALCULO ---
    df_det, df_res = calcular_escenario(df_bbdd_raw, df_real_raw, params)

    # --- VISUALIZACION ---
    col1, col2, col3 = st.columns(3)
    for i, pais in enumerate(["Chile", "Colombia", "Perú"]):
        row = df_res[df_res["Pais"]==pais]
        val = row["Var_%"].values[0] if not row.empty else 0
        delta = row["Acceso_Proyectado"].values[0] - row["Acceso_Actual"].values[0] if not row.empty else 0
        with [col1, col2, col3][i]:
            st.metric(f"{pais} Var %", f"{val:+.1f}%", f"${delta:,.0f}")

    st.subheader("Comparativo Actual (Hoja Negociación) vs Propuesto (BBDD simulada)")
    st.bar_chart(df_res.set_index("Pais")[["Acceso_Actual", "Acceso_Proyectado"]], color=["#999999", "#FF4B4B"])
    
    st.dataframe(df_det.style.format("${:,.0f}", subset=["Monto_USD", "Acceso_Actual", "Acceso_Proyectado"]), use_container_width=True)