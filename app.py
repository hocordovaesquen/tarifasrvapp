import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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
2. Ajusta las tarifas Fijas y Variables en el menú de la izquierda.
3. El sistema recalculará los ingresos proyectados vs. los reales del archivo.
""")

# ---------------------------------------------------------
# Funciones de Carga y Limpieza (Backend)
# ---------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list) -> str:
    """
    Busca una columna en el DataFrame probando una lista de nombres posibles.
    Retorna el nombre real encontrado en el df.
    """
    # Crear mapa de columnas en minúsculas y sin espacios para búsqueda flexible
    norm = {str(c).strip().lower(): c for c in df.columns}
    
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
            
    # Si no encuentra, retorna el primer candidato para que falle con un error más claro luego
    return candidates[0] 

def load_bbdd_from_bytes(data: bytes, sheet_name: str = "A.3 BBDD Neg") -> pd.DataFrame:
    """
    Carga la hoja de base de datos buscando dinámicamente dónde empieza el encabezado.
    """
    try:
        # 1. Leemos las primeras 20 filas sin encabezado para "escanear"
        raw_preview = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=None, nrows=20)
        
        header_idx = -1
        # 2. Buscamos la fila que contenga palabras clave como "Pais" y "Corredor"
        for i, row in raw_preview.iterrows():
            row_str = row.astype(str).str.lower().values.tolist()
            if any("pais" in s for s in row_str) and any("corredor" in s for s in row_str):
                header_idx = i
                break
        
        # Fallback: Si no encuentra nada, asumimos fila 7 (común en tus archivos)
        if header_idx == -1:
            header_idx = 7

        # 3. Cargamos el dataframe real usando la fila detectada como header
        df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=header_idx)
        
        # Limpieza de nombres de columnas (eliminar espacios al inicio/final)
        df.columns = [str(c).strip() for c in df.columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error leyendo la hoja '{sheet_name}'. Asegúrate de que existe en el Excel.\nDetalle: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_excel_data(excel_bytes: bytes):
    """
    Función principal de carga.
    """
    df_base = load_bbdd_from_bytes(excel_bytes, sheet_name="A.3 BBDD Neg")
    return df_base

# ---------------------------------------------------------
# Lógica de Negocio (Cálculo Financiero)
# ---------------------------------------------------------

def calcular_escenario_acceso(df_base: pd.DataFrame, params_por_pais: dict):
    """
    Realiza la simulación de ingresos.
    
    Lógica:
    1. Agrupa los datos por PAIS y BROKER (para no duplicar cobros fijos).
    2. Suma el 'Monto USD' y el 'Acceso Actual' (Real).
    3. Calcula 'Acceso Proyectado' = (Volumen * Tasa Variable) + (Tarifa Fija * 12).
    """
    df = df_base.copy()
    
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    # 1. Identificar columnas dinámicamente
    col_pais = find_column(df, ["Pais", "País", "Country"])
    col_broker = find_column(df, ["Corredor", "Broker", "Cliente estandar", "Cliente"])
    col_monto = find_column(df, ["Monto USD", "Monto_USD", "Volumen USD", "Volumen"])
    col_acceso_actual = find_column(df, ["Cobro Acceso", "Acceso actual", "Total ingresos", "Ingreso Actual"])

    # 2. Renombrar para estandarizar
    df.rename(columns={
        col_pais: "Pais",
        col_broker: "Broker",
        col_monto: "Monto_USD",
        col_acceso_actual: "Acceso_Actual"
    }, inplace=True)

    # 3. Limpieza de tipos de datos
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors='coerce').fillna(0.0)
    df["Acceso_Actual"] = pd.to_numeric(df["Acceso_Actual"], errors='coerce').fillna(0.0)
    
    # Normalizar nombres de países
    df["Pais"] = df["Pais"].astype(str).str.upper().str.strip()
    df["Pais"] = df["Pais"].replace({
        "PERU": "Perú", "PERÚ": "Perú", 
        "CHILE": "Chile", 
        "COLOMBIA": "Colombia"
    })
    
    # Filtrar solo los 3 países de interés
    df = df[df["Pais"].isin(["Chile", "Colombia", "Perú", "CHILE", "COLOMBIA", "PERU"])]

    # -------------------------------------------------------
    # 4. AGRUPACIÓN (El paso clave para corregir los números)
    # -------------------------------------------------------
    # Agrupamos por Broker para tener UNA fila por cliente
    df_grouped = df.groupby(["Pais", "Broker"]).agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum"
    }).reset_index()

    # 5. Aplicar Tarifas Proyectadas
    proyectado_list = []
    
    for idx, row in df_grouped.iterrows():
        pais = row["Pais"]
        monto = row["Monto_USD"]
        
        # Obtener parámetros del slider
        cfg = params_por_pais.get(pais, {"fijo": 0, "bps": 0})
        fijo_mensual = cfg.get("fijo", 0.0)
        bps_variable = cfg.get("bps", 0.0)

        # FÓRMULA:
        # Variable: (Monto Anual * bps) / 10,000
        ingreso_variable = monto * (bps_variable / 10000.0)
        
        # Fijo: Tarifa Mensual * 12 meses
        ingreso_fijo = fijo_mensual * 12.0
        
        total_proyectado = ingreso_variable + ingreso_fijo
        proyectado_list.append(total_proyectado)

    df_grouped["Acceso_Proyectado"] = proyectado_list

    # 6. Calcular Variaciones
    df_grouped["Var_Abs"] = df_grouped["Acceso_Proyectado"] - df_grouped["Acceso_Actual"]
    df_grouped["Var_%"] = np.where(
        df_grouped["Acceso_Actual"] > 50, # Evitar división por 0 o ruido en montos chicos
        (df_grouped["Var_Abs"] / df_grouped["Acceso_Actual"]) * 100,
        0.0
    )

    # 7. Crear DataFrame Resumen por País (para KPIs)
    agg_pais = df_grouped.groupby("Pais").agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()

    # BPS Implícitos
    agg_pais["BPS_Proyectado"] = (agg_pais["Acceso_Proyectado"] / agg_pais["Monto_USD"]) * 10000
    agg_pais["Var_%"] = ((agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]) / agg_pais["Acceso_Actual"]) * 100
    
    return df_grouped, agg_pais

# ---------------------------------------------------------
# Interfaz de Usuario (Streamlit)
# ---------------------------------------------------------

# 1. Uploader
uploaded_file = st.file_uploader("📥 Cargar Excel (Modelamiento...)", type=["xlsx"])

if not uploaded_file:
    st.info("Esperando archivo...")
    st.stop()

# Cargar datos
with st.spinner("Procesando Excel..."):
    df_base = load_excel_data(uploaded_file.getvalue())

if df_base.empty:
    st.error("El archivo no contiene datos válidos en la hoja 'A.3 BBDD Neg'.")
    st.stop()

st.success(f"Datos cargados: {len(df_base)} registros encontrados.")

# 2. Sidebar de Parámetros
st.sidebar.header("⚙️ Configuración de Tarifas")
st.sidebar.info("Nota: La tarifa fija se multiplicará por 12 (anualizado).")

params = {}
paises = ["Chile", "Colombia", "Perú"]

# Valores por defecto para iniciar
defaults = {
    "Chile": {"fijo": 500.0, "bps": 0.15},
    "Colombia": {"fijo": 400.0, "bps": 0.20},
    "Perú": {"fijo": 350.0, "bps": 0.25}
}

for pais in paises:
    st.sidebar.markdown(f"### 🇨🇱 🇨🇴 🇵🇪 {pais}")
    fijo = st.sidebar.number_input(
        f"Tarifa Fija Mensual (USD) - {pais}", 
        min_value=0.0, 
        value=defaults[pais]["fijo"], 
        step=50.0
    )
    bps = st.sidebar.number_input(
        f"Tarifa Variable (bps) - {pais}", 
        min_value=0.0, 
        value=defaults[pais]["bps"], 
        step=0.05,
        format="%.2f"
    )
    params[pais] = {"fijo": fijo, "bps": bps}

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recalcular Todo"):
    pass # El script se re-ejecuta solo al cambiar inputs, el botón es visual

# 3. Ejecutar Cálculos
df_detalle, df_resumen = calcular_escenario_acceso(df_base, params)

# 4. Mostrar KPIs (Tarjetas Superiores)
st.subheader("🏁 Impacto General por País")
cols = st.columns(3)

for idx, pais in enumerate(paises):
    # Buscar datos del país en el resumen
    row = df_resumen[df_resumen["Pais"] == pais]
    
    if not row.empty:
        var_pct = row["Var_%"].values[0]
        bps_proj = row["BPS_Proyectado"].values[0]
        ingreso_new = row["Acceso_Proyectado"].values[0]
        ingreso_old = row["Acceso_Actual"].values[0]
        delta = ingreso_new - ingreso_old
    else:
        var_pct = 0; bps_proj = 0; ingreso_new = 0; delta = 0

    with cols[idx]:
        st.metric(
            label=f"{pais} - Var. Ingresos",
            value=f"{var_pct:+.1f}%",
            delta=f"${delta:,.0f} USD",
            delta_color="normal"
        )
        st.caption(f"BPS Proyectado: **{bps_proj:.2f}** bps")

# 5. Gráficos y Tablas
st.markdown("---")
tab1, tab2 = st.tabs(["📊 Gráfico Comparativo", "📋 Detalle por Broker"])

with tab1:
    st.subheader("Ingresos: Actual vs Proyectado (USD)")
    
    # Preparamos datos para gráfico de barras
    if not df_resumen.empty:
        chart_data = df_resumen[["Pais", "Acceso_Actual", "Acceso_Proyectado"]].set_index("Pais")
        st.bar_chart(chart_data, color=["#A9A9A9", "#FF4B4B"]) # Gris para actual, Rojo para nuevo
        st.caption("Gris: Ingreso Actual | Rojo: Ingreso Proyectado")
    else:
        st.warning("No hay datos para graficar.")

with tab2:
    st.subheader("Detalle Cliente por Cliente")
    
    # Filtros de visualización
    filtro_pais = st.selectbox("Filtrar por País:", ["Todos"] + paises)
    
    view_df = df_detalle.copy()
    if filtro_pais != "Todos":
        view_df = view_df[view_df["Pais"] == filtro_pais]
        
    # Formateo para mostrar en pantalla
    st.dataframe(
        view_df.style.format({
            "Monto_USD": "${:,.0f}",
            "Acceso_Actual": "${:,.0f}",
            "Acceso_Proyectado": "${:,.0f}",
            "Var_%": "{:+.1f}%",
            "Var_Abs": "${:,.0f}"
        }),
        use_container_width=True,
        height=500
    )

    # Botón de descarga
    csv = view_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar Detalle en CSV",
        data=csv,
        file_name="simulacion_tarifas_nuam.csv",
        mime="text/csv"
    )