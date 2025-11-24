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
2. Configura los **Tramos Mensuales** en el menú de la izquierda.
3. El sistema comparará lo que pagaron realmente (según tu BBDD) vs. la nueva propuesta calculada mes a mes.
""")

# ---------------------------------------------------------
# Funciones de Carga y Limpieza (Backend)
# ---------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list) -> str:
    """
    Busca una columna en el DataFrame probando una lista de nombres posibles.
    Retorna el nombre real encontrado en el df.
    """
    norm = {str(c).strip().lower(): c for c in df.columns}
    
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
            
    return candidates[0] # Retorno por defecto si falla

def load_bbdd_from_bytes(data: bytes, sheet_name: str = "A.3 BBDD Neg") -> pd.DataFrame:
    """
    Carga la hoja de base de datos buscando dinámicamente dónde empieza el encabezado.
    """
    try:
        # 1. Escaneo previo para encontrar la fila de títulos
        raw_preview = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=None, nrows=20)
        
        header_idx = -1
        for i, row in raw_preview.iterrows():
            row_str = row.astype(str).str.lower().values.tolist()
            # Buscamos filas que tengan "pais" Y "corredor" (o "cliente")
            if any("pais" in s for s in row_str) and (any("corredor" in s for s in row_str) or any("cliente" in s for s in row_str)):
                header_idx = i
                break
        
        if header_idx == -1:
            header_idx = 7 # Fallback estándar de tus archivos

        # 2. Carga definitiva
        df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=header_idx)
        df.columns = [str(c).strip() for c in df.columns] # Limpiar espacios en nombres
        
        return df
        
    except Exception as e:
        st.error(f"Error leyendo la hoja '{sheet_name}'. Verifica que el archivo sea el correcto.\nDetalle: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_excel_data(excel_bytes: bytes):
    df_base = load_bbdd_from_bytes(excel_bytes, sheet_name="A.3 BBDD Neg")
    return df_base

# ---------------------------------------------------------
# Lógica de Negocio (Cálculo Mensual Escalonado)
# ---------------------------------------------------------

def calcular_escenario_acceso(df_base: pd.DataFrame, params_por_pais: dict):
    """
    Simulación basada en TRAMOS MENSUALES.
    
    Lógica:
    1. Agrupar datos por [Pais, Broker, MES].
    2. Sumar Volumen USD del mes.
    3. Sumar 'Cobro Acceso' (Actual) del mes (esto viene fijo de la BBDD).
    4. Calcular 'Acceso Proyectado' aplicando la tabla de tramos al volumen del mes.
    5. Consolidar anualmente.
    """
    df = df_base.copy()
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. Mapeo de columnas
    col_pais = find_column(df, ["Pais", "País", "Country"])
    col_broker = find_column(df, ["Corredor", "Broker", "Cliente estandar", "Cliente"])
    col_monto = find_column(df, ["Monto USD", "Monto_USD", "Volumen USD"])
    col_mes = find_column(df, ["Mes", "Month", "Month_ID"]) 
    col_acceso_actual = find_column(df, ["Cobro Acceso", "Acceso actual", "Total ingresos", "Ingreso Actual"])

    # Renombrar para uso interno
    df.rename(columns={
        col_pais: "Pais",
        col_broker: "Broker",
        col_monto: "Monto_USD",
        col_mes: "Mes",
        col_acceso_actual: "Acceso_Actual"
    }, inplace=True)

    # Limpieza numérica
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors='coerce').fillna(0.0)
    df["Acceso_Actual"] = pd.to_numeric(df["Acceso_Actual"], errors='coerce').fillna(0.0)
    
    # Normalizar País
    df["Pais"] = df["Pais"].astype(str).str.upper().str.strip()
    df["Pais"] = df["Pais"].replace({
        "PERU": "Perú", "PERÚ": "Perú", 
        "CHILE": "Chile", "COLOMBIA": "Colombia"
    })
    
    # Filtrar solo nuam
    df = df[df["Pais"].isin(["Chile", "Colombia", "Perú"])]

    # -------------------------------------------------------------------------
    # PASO CLAVE: Agrupación Mensual
    # -------------------------------------------------------------------------
    # Primero sumamos todas las operaciones dentro del mismo mes para cada broker
    df_mensual = df.groupby(["Pais", "Broker", "Mes"]).agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum" # Mantenemos la verdad histórica
    }).reset_index()

    # Función para aplicar tramo
    def aplicar_tarifa_mes(row):
        pais = row["Pais"]
        volumen = row["Monto_USD"]
        
        cfg = params_por_pais.get(pais)
        if not cfg: return 0.0

        # Lógica de Tramos (Tiered)
        if volumen < cfg["limite_bajo"]:
            return cfg["tarifa_bajo"]
        elif volumen < cfg["limite_alto"]:
            return cfg["tarifa_medio"]
        else:
            return cfg["tarifa_alto"]

    # Calculamos la cuota de ESE mes
    df_mensual["Acceso_Proyectado_Mes"] = df_mensual.apply(aplicar_tarifa_mes, axis=1)

    # -------------------------------------------------------------------------
    # Consolidación Anual (Para Reporte)
    # -------------------------------------------------------------------------
    df_anual = df_mensual.groupby(["Pais", "Broker"]).agg({
        "Monto_USD": "sum",             # Volumen total año
        "Acceso_Actual": "sum",         # Total pagado en el año (Histórico)
        "Acceso_Proyectado_Mes": "sum"  # Suma de las 12 cuotas simuladas
    }).reset_index()

    df_anual.rename(columns={"Acceso_Proyectado_Mes": "Acceso_Proyectado"}, inplace=True)

    # Cálculos de Variación
    df_anual["Var_Abs"] = df_anual["Acceso_Proyectado"] - df_anual["Acceso_Actual"]
    
    # Evitamos división por cero o ruido visual en variaciones
    df_anual["Var_%"] = np.where(
        df_anual["Acceso_Actual"] > 50, 
        (df_anual["Var_Abs"] / df_anual["Acceso_Actual"]) * 100,
        0.0 
    )

    # Resumen por País
    agg_pais = df_anual.groupby("Pais").agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()

    agg_pais["BPS_Proyectado"] = (agg_pais["Acceso_Proyectado"] / agg_pais["Monto_USD"]) * 10000
    agg_pais["Var_%"] = ((agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]) / agg_pais["Acceso_Actual"]) * 100
    
    return df_anual, agg_pais

# ---------------------------------------------------------
# Interfaz de Usuario (Streamlit)
# ---------------------------------------------------------

# 1. Upload
uploaded_file = st.file_uploader("📥 Cargar Excel (Modelamiento...)", type=["xlsx"])

if not uploaded_file:
    st.info("Esperando archivo...")
    st.stop()

with st.spinner("Procesando Excel..."):
    df_base = load_excel_data(uploaded_file.getvalue())

if df_base.empty:
    st.error("No se encontraron datos. Revisa la hoja 'A.3 BBDD Neg'.")
    st.stop()

st.success(f"Datos cargados exitosamente: {len(df_base)} registros procesados.")

# 2. Sidebar: Configuración de Tramos
st.sidebar.header("⚙️ Configuración de Tramos (Mensual)")
st.sidebar.info("Define los rangos de volumen mensual (en MM USD) y la tarifa fija para ese tramo.")

params = {}
paises = ["Chile", "Colombia", "Perú"]

# Valores iniciales sugeridos
defaults = {
    "Chile":    {"L1": 5.0, "L2": 15.0, "T1": 500.0, "T2": 1500.0, "T3": 2500.0},
    "Colombia": {"L1": 5.0, "L2": 15.0, "T1": 400.0, "T2": 1200.0, "T3": 2000.0},
    "Perú":     {"L1": 5.0, "L2": 15.0, "T1": 350.0, "T2": 1500.0, "T3": 2000.0}
}

for pais in paises:
    with st.sidebar.expander(f"Configurar {pais}", expanded=False):
        st.markdown("**Límites de Volumen (Millones USD)**")
        l1 = st.number_input(f"Límite Tramo 1 (MM) - {pais}", value=defaults[pais]["L1"], step=1.0)
        l2 = st.number_input(f"Límite Tramo 2 (MM) - {pais}", value=defaults[pais]["L2"], step=1.0)
        
        st.markdown("**Tarifas Mensuales (USD)**")
        t1 = st.number_input(f"Tarifa Baja (< {l1}M) - {pais}", value=defaults[pais]["T1"], step=50.0)
        t2 = st.number_input(f"Tarifa Media ({l1}-{l2}M) - {pais}", value=defaults[pais]["T2"], step=50.0)
        t3 = st.number_input(f"Tarifa Alta (> {l2}M) - {pais}", value=defaults[pais]["T3"], step=50.0)

        params[pais] = {
            "limite_bajo": l1 * 1_000_000, 
            "limite_alto": l2 * 1_000_000,
            "tarifa_bajo": t1,
            "tarifa_medio": t2,
            "tarifa_alto": t3
        }

if st.sidebar.button("🔄 Recalcular"):
    pass 

# 3. Ejecución
df_detalle, df_resumen = calcular_escenario_acceso(df_base, params)

# 4. KPIs
st.subheader("🏁 Impacto General (Anualizado)")
cols = st.columns(3)

for idx, pais in enumerate(paises):
    row = df_resumen[df_resumen["Pais"] == pais]
    if not row.empty:
        var_pct = row["Var_%"].values[0]
        ingreso_new = row["Acceso_Proyectado"].values[0]
        ingreso_old = row["Acceso_Actual"].values[0]
        delta = ingreso_new - ingreso_old
    else:
        var_pct = 0; delta = 0; ingreso_new = 0

    with cols[idx]:
        st.metric(
            label=f"{pais} - Var. Ingresos",
            value=f"{var_pct:+.1f}%",
            delta=f"${delta:,.0f} USD",
            delta_color="normal"
        )
        st.caption(f"Proyectado: ${ingreso_new:,.0f} | Actual: ${ingreso_old:,.0f}")

st.markdown("---")

# 5. Gráfico y Tabla
tab1, tab2 = st.tabs(["📊 Gráfico Comparativo", "📋 Detalle por Broker"])

with tab1:
    st.subheader("Ingresos Totales: Actual vs Proyectado")
    if not df_resumen.empty:
        chart_data = df_resumen[["Pais", "Acceso_Actual", "Acceso_Proyectado"]].set_index("Pais")
        st.bar_chart(chart_data, color=["#A9A9A9", "#FF4B4B"]) 
        st.caption("Gris: Real (BBDD) | Rojo: Simulado (Tramos)")
    else:
        st.warning("No hay datos para mostrar.")

with tab2:
    st.subheader("Detalle Anual por Broker")
    filtro_pais = st.selectbox("Filtrar por País:", ["Todos"] + paises)
    
    view_df = df_detalle.copy()
    if filtro_pais != "Todos":
        view_df = view_df[view_df["Pais"] == filtro_pais]
        
    st.dataframe(
        view_df.style.format({
            "Monto_USD": "${:,.0f}",
            "Acceso_Actual": "${:,.0f}",
            "Acceso_Proyectado": "${:,.0f}",
            "Var_%": "{:+.1f}%",
            "Var_Abs": "${:,.0f}"
        }),
        use_container_width=True,
        height=600
    )
    
    # Botón descarga
    csv = view_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar Resultados CSV",
        data=csv,
        file_name="simulacion_acceso_nuam.csv",
        mime="text/csv"
    )