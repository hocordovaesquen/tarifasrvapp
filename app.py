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
**Lógica del Sistema:**
1. **Datos Reales:** Se extraen de la hoja `3. Negociación` (Fila 10+, Columna C=Broker, Columna I=Acceso).
2. **Simulación:** Se calcula mes a mes usando los volúmenes de `A.3 BBDD Neg` y los tramos que configures.
""")

# ---------------------------------------------------------
# Funciones de Limpieza y Normalización
# ---------------------------------------------------------

def clean_broker_name(name):
    """
    Normaliza los nombres de los brokers para asegurar el cruce entre hojas.
    Ej: "Kallpa SAB S.A." -> "KALLPA"
    """
    if pd.isna(name): return ""
    name = str(name).upper().strip()
    
    # Palabras a eliminar para estandarizar
    remove_words = [
        "S.A.", "S.A", "S.A.B.", "SAB", "CORREDORES", "DE BOLSA", 
        "SOCIEDAD AGENTE", "VALORES", "COMISIONISTA", "S.C.B.", 
        "SPA", "LTDA", "AGENTE", "BANCO", "CORREDORA"
    ]
    
    for word in remove_words:
        name = name.replace(word, "")
    
    # Quitar caracteres especiales y espacios dobles
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Casos especiales manuales (Diccionario de alias)
    alias = {
        "BTG PACTUAL CHILE": "BTG PACTUAL",
        "BTG PACTUAL PERU": "BTG PACTUAL",
        "CREDICORP CAPITAL": "CREDICORP",
        "LARRAIN VIAL": "LARRAINVIAL",
        "SCOTIA": "SCOTIABANK"
    }
    
    for k, v in alias.items():
        if k in name:
            return v
            
    return name

# ---------------------------------------------------------
# Carga de Datos (Surgical Loading)
# ---------------------------------------------------------

def load_data_complete(excel_bytes):
    xls = pd.ExcelFile(BytesIO(excel_bytes))
    
    # --- 1. CARGA DE REALES (Hoja "3. Negociación") ---
    # Instrucción: Fila 10 en adelante. Col C (Idx 2) y Col I (Idx 8).
    sheet_real = "3. Negociación"
    df_real = pd.DataFrame()
    
    if sheet_real in xls.sheet_names:
        # header=None para controlar índices manualmente. skiprows=9 para empezar en fila 10.
        raw_real = pd.read_excel(xls, sheet_name=sheet_real, header=None, skiprows=9)
        
        # Verificar que tenga suficientes columnas
        if raw_real.shape[1] >= 9:
            # Seleccionamos Columna C (2) y Columna I (8)
            df_real = raw_real.iloc[:, [2, 8]].copy()
            df_real.columns = ["Broker_Raw", "Acceso_Actual"]
            
            # Limpieza
            df_real = df_real.dropna(subset=["Broker_Raw"]) # Borrar filas sin nombre
            df_real["Acceso_Actual"] = pd.to_numeric(df_real["Acceso_Actual"], errors='coerce').fillna(0)
            
            # Generar llave de cruce
            df_real["Broker_Key"] = df_real["Broker_Raw"].apply(clean_broker_name)
            
            # Agrupar por si hay duplicados por limpieza de nombres
            df_real = df_real.groupby("Broker_Key")["Acceso_Actual"].sum().reset_index()
        else:
            st.error(f"La hoja '{sheet_real}' no tiene suficientes columnas (se esperaba hasta la I).")
    else:
        st.error(f"No se encontró la hoja '{sheet_real}'.")

    # --- 2. CARGA DE BBDD (Hoja "A.3 BBDD Neg") ---
    sheet_bbdd = "A.3 BBDD Neg"
    df_bbdd = pd.DataFrame()
    
    if sheet_bbdd in xls.sheet_names:
        # Buscamos el header dinámicamente como antes
        preview = pd.read_excel(xls, sheet_name=sheet_bbdd, header=None, nrows=20)
        idx = 7 # Default
        for i, row in preview.iterrows():
            row_s = row.astype(str).str.lower().values
            if any("pais" in s for s in row_s) and any("corredor" in s for s in row_s):
                idx = i; break
        
        df_bbdd = pd.read_excel(xls, sheet_name=sheet_bbdd, header=idx)
        df_bbdd.columns = [str(c).strip() for c in df_bbdd.columns]
    else:
        st.error(f"No se encontró la hoja '{sheet_bbdd}'.")

    return df_bbdd, df_real

# ---------------------------------------------------------
# Lógica de Cálculo (Simulación + Cruce)
# ---------------------------------------------------------

def calcular_simulacion(df_bbdd, df_real_clean, params_tramos):
    """
    1. Calcula Proyección mensual usando df_bbdd.
    2. Cruza con df_real_clean para traer el dato 'Actual'.
    """
    if df_bbdd.empty: return pd.DataFrame(), pd.DataFrame()

    df = df_bbdd.copy()
    
    # 1. Identificar columnas BBDD
    col_map = {c.lower(): c for c in df.columns}
    # Buscamos nombres probables
    c_pais = next((col_map[k] for k in col_map if "pais" in k or "país" in k), "Pais")
    c_broker = next((col_map[k] for k in col_map if "corredor" in k or "broker" in k or "cliente" in k), "Corredor")
    c_monto = next((col_map[k] for k in col_map if "monto usd" in k or "volumen usd" in k), "Monto USD")
    c_mes = next((col_map[k] for k in col_map if "mes" in k or "month" in k), "Mes")

    # Renombrar para estandarizar código
    df.rename(columns={c_pais:"Pais", c_broker:"Broker", c_monto:"Monto_USD", c_mes:"Mes"}, inplace=True)
    
    # Filtros y Tipos
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors='coerce').fillna(0)
    df["Pais"] = df["Pais"].astype(str).str.upper().str.strip().replace({
        "PERU": "Perú", "PERÚ": "Perú", "CHILE": "Chile", "COLOMBIA": "Colombia"
    })
    df = df[df["Pais"].isin(["Chile", "Colombia", "Perú"])]

    # -------------------------------------------------------
    # PASO A: CALCULAR PROYECTADO (Mes a Mes)
    # -------------------------------------------------------
    # Agrupamos por mes para aplicar tramos
    df_mensual = df.groupby(["Pais", "Broker", "Mes"])["Monto_USD"].sum().reset_index()

    def aplicar_tramos(row):
        p_cfg = params_tramos.get(row["Pais"])
        if not p_cfg: return 0.0
        
        vol = row["Monto_USD"]
        # Lógica de Tramos
        if vol < p_cfg["L1"]: return p_cfg["T1"]
        elif vol < p_cfg["L2"]: return p_cfg["T2"]
        else: return p_cfg["T3"]

    df_mensual["Acceso_Proyectado"] = df_mensual.apply(aplicar_tramos, axis=1)

    # Consolidar Anual por Broker
    df_final = df_mensual.groupby(["Pais", "Broker"]).agg({
        "Monto_USD": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()

    # -------------------------------------------------------
    # PASO B: CRUZAR CON REAL ("3. Negociación")
    # -------------------------------------------------------
    if not df_real_clean.empty:
        # Creamos llave en df_final
        df_final["Broker_Key"] = df_final["Broker"].apply(clean_broker_name)
        
        # Merge Left (Mantenemos todos los brokers de la simulación)
        df_merged = pd.merge(df_final, df_real_clean, on="Broker_Key", how="left")
        
        # Llenar nulos con 0 (Si no está en hoja Negociación, asumimos 0)
        df_merged["Acceso_Actual"] = df_merged["Acceso_Actual"].fillna(0)
        
        # Limpiar
        df_final = df_merged.drop(columns=["Broker_Key"])
    else:
        df_final["Acceso_Actual"] = 0.0

    # -------------------------------------------------------
    # PASO C: KPIs Finales
    # -------------------------------------------------------
    df_final["Var_Abs"] = df_final["Acceso_Proyectado"] - df_final["Acceso_Actual"]
    df_final["Var_%"] = np.where(
        df_final["Acceso_Actual"] > 50, # Evitar división por 0
        (df_final["Var_Abs"] / df_final["Acceso_Actual"]) * 100,
        0.0
    )

    # Resumen País
    agg_pais = df_final.groupby("Pais").agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()
    
    agg_pais["Var_%"] = np.where(
        agg_pais["Acceso_Actual"] > 0,
        ((agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]) / agg_pais["Acceso_Actual"]) * 100,
        0.0
    )

    return df_final, agg_pais

# ---------------------------------------------------------
# Interfaz de Usuario
# ---------------------------------------------------------

uploaded_file = st.file_uploader("📥 Cargar Excel 'Modelamiento Estructura Tarifaria'", type=["xlsx"])

if uploaded_file:
    with st.spinner("Procesando hojas..."):
        df_bbdd_raw, df_real_clean = load_data_complete(uploaded_file.getvalue())

    if df_bbdd_raw.empty:
        st.error("Error: No se pudieron cargar los datos de la BBDD.")
        st.stop()

    st.success(f"✅ BBDD Cargada. ✅ Datos Reales extraídos de hoja '3. Negociación' ({len(df_real_clean)} brokers encontrados).")

    # --- SIDEBAR: TRAMOS ---
    st.sidebar.header("⚙️ Configuración de Tramos")
    st.sidebar.caption("Define límites mensuales (MM USD) y tarifas (USD).")
    
    params = {}
    # Valores default según conversación
    defaults = {
        "Chile":    {"L1":5.0, "L2":15.0, "T1":500.0, "T2":1500.0, "T3":2500.0},
        "Colombia": {"L1":5.0, "L2":15.0, "T1":400.0, "T2":1200.0, "T3":2000.0},
        "Perú":     {"L1":5.0, "L2":15.0, "T1":350.0, "T2":1500.0, "T3":2000.0}
    }
    
    for pais in ["Chile", "Colombia", "Perú"]:
        with st.sidebar.expander(f"{pais}", expanded=False):
            # Límites en Millones
            l1 = st.number_input(f"Límite 1 (MM) {pais}", value=defaults[pais]["L1"])
            l2 = st.number_input(f"Límite 2 (MM) {pais}", value=defaults[pais]["L2"])
            # Tarifas
            t1 = st.number_input(f"Tarifa Baja (<{l1}M) {pais}", value=defaults[pais]["T1"])
            t2 = st.number_input(f"Tarifa Media {pais}", value=defaults[pais]["T2"])
            t3 = st.number_input(f"Tarifa Alta (>{l2}M) {pais}", value=defaults[pais]["T3"])
            
            # Guardamos (convertimos límites a unidades completas)
            params[pais] = {
                "L1": l1 * 1_000_000, "L2": l2 * 1_000_000,
                "T1": t1, "T2": t2, "T3": t3
            }

    # --- CÁLCULOS ---
    df_detalle, df_resumen = calcular_simulacion(df_bbdd_raw, df_real_clean, params)

    # --- RESULTADOS ---
    st.subheader("🏁 Resumen de Impacto (Anualizado)")
    
    cols = st.columns(3)
    for i, pais in enumerate(["Chile", "Colombia", "Perú"]):
        row = df_resumen[df_resumen["Pais"] == pais]
        if not row.empty:
            val = row["Var_%"].values[0]
            delta = row["Acceso_Proyectado"].values[0] - row["Acceso_Actual"].values[0]
            old = row["Acceso_Actual"].values[0]
            new = row["Acceso_Proyectado"].values[0]
        else:
            val=0; delta=0; old=0; new=0
            
        with cols[i]:
            st.metric(f"{pais} Var %", f"{val:+.1f}%", f"${delta:,.0f}")
            st.caption(f"Actual: ${old:,.0f} | Propuesto: ${new:,.0f}")

    # Gráfico
    st.markdown("---")
    st.subheader("Comparativo de Ingresos")
    if not df_resumen.empty:
        chart_data = df_resumen.set_index("Pais")[["Acceso_Actual", "Acceso_Proyectado"]]
        st.bar_chart(chart_data, color=["#A9A9A9", "#FF4B4B"])
        st.caption("Gris: Real (Hoja Negociación) | Rojo: Simulado")

    # Tabla Detalle
    st.subheader("Detalle por Broker")
    pais_filter = st.selectbox("Filtrar país", ["Todos", "Chile", "Colombia", "Perú"])
    
    view = df_detalle.copy()
    if pais_filter != "Todos":
        view = view[view["Pais"] == pais_filter]
        
    st.dataframe(
        view.style.format({
            "Monto_USD": "${:,.0f}",
            "Acceso_Actual": "${:,.0f}",
            "Acceso_Proyectado": "${:,.0f}",
            "Var_%": "{:+.1f}%",
            "Var_Abs": "${:,.0f}"
        }),
        use_container_width=True,
        height=600
    )