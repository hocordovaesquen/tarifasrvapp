import streamlit as st
import pandas as pd

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Tarifario RV", layout="wide")

st.title("📊 Simulador de Estructura Tarifaria RV")
st.markdown("""
Esta aplicación reemplaza el cálculo manual de Excel. 
**Instrucciones:** Sube tu archivo `.xlsx` completo y ajusta los parámetros en la barra lateral para simular escenarios.
""")

# --- 2. BARRA LATERAL: CONTROL DE PARÁMETROS (Inputs) ---
st.sidebar.header("1. Variables de Simulación")

st.sidebar.subheader("Macroeconomía (Tasas FX)")
# Valores por defecto tomados de tu hoja '1. Parametros'
tasa_cop = st.sidebar.number_input("Tasa COP/USD", value=4066.60, step=10.0)
tasa_clp = st.sidebar.number_input("Tasa CLP/USD", value=907.20, step=5.0)
tasa_pen = st.sidebar.number_input("Tasa PEN/USD", value=3.75, step=0.05)

# Diccionario para mapear monedas automáticamente
tasas_cambio = {
    "COP": tasa_cop,
    "CLP": tasa_clp,
    "PEN": tasa_pen,
    "USD": 1.0
}

st.sidebar.subheader("Estructura de Tarifas (BPS)")
st.sidebar.info("Define los tramos y tarifas a aplicar sobre el Volumen USD.")

# Definición dinámica de tramos
limite_tramo_1 = st.sidebar.number_input("Límite Tramo 1 (USD)", value=250_000_000, step=1_000_000)
limite_tramo_2 = st.sidebar.number_input("Límite Tramo 2 (USD)", value=500_000_000, step=1_000_000)

tarifa_t1 = st.sidebar.slider("Tarifa Tramo 1 (bps)", 0.0, 2.0, 0.60)
tarifa_t2 = st.sidebar.slider("Tarifa Tramo 2 (bps)", 0.0, 2.0, 0.50)
tarifa_t3 = st.sidebar.slider("Tarifa Tramo 3 (> Límite 2)", 0.0, 2.0, 0.40)

# --- 3. LÓGICA DE NEGOCIO (EL MOTOR) ---
def calcular_tarifa_bps(volumen_usd):
    """Determina el BPS a cobrar según el volumen negociado"""
    if volumen_usd <= limite_tramo_1:
        return tarifa_t1
    elif volumen_usd <= limite_tramo_2:
        return tarifa_t2
    else:
        return tarifa_t3

def convertir_moneda(monto, moneda_origen):
    """Convierte a USD usando las tasas del sidebar"""
    # Normalizamos el texto (ej: 'COP ' -> 'COP')
    moneda_clean = str(moneda_origen).strip().upper()
    tasa = tasas_cambio.get(moneda_clean, 1.0) # Si no encuentra, asume 1 a 1
    if tasa == 0: return 0
    # Para la mayoría de monedas LATAM la lógica es Dividir (COP/Tasa)
    return monto / tasa

# --- 4. INTERFAZ PRINCIPAL ---
st.header("2. Carga de Datos")
uploaded_file = st.file_uploader("Sube tu archivo Excel 'Modelamiento Estructura Tarifaria'", type=["xlsx"])

if uploaded_file:
    try:
        # Leemos el archivo Excel en memoria
        xls = pd.ExcelFile(uploaded_file)
        
        # Verificamos si existe la hoja de transacciones
        if 'A.3 BBDD Neg' in xls.sheet_names:
            
            # --- LECTURA Y LIMPIEZA DE DATOS ---
            # header=7 significa que empieza a leer en la fila 8 (donde están los títulos reales en tu Excel)
            df = pd.read_excel(xls, sheet_name='A.3 BBDD Neg', header=7)
            
            # Filtramos filas basura (donde no hay Corredor o Monto)
            df_clean = df.dropna(subset=['Corredor', 'Monto Local']).copy()
            
            st.write(f"✅ Se cargaron **{len(df_clean)}** transacciones procesables.")
            
            # Mostramos una muestra cruda
            with st.expander("Ver datos originales cargados"):
                st.dataframe(df_clean.head())

            if st.button("🚀 Ejecutar Simulación End-to-End"):
                
                # --- PROCESAMIENTO VECTORIAL (RÁPIDO) ---
                
                # 1. Convertir a USD
                # Usamos una función lambda para aplicar la conversión fila por fila
                df_clean['Volumen_USD_Simulado'] = df_clean.apply(
                    lambda row: convertir_moneda(row['Monto Local'], row['Moneda']), axis=1
                )
                
                # 2. Calcular Tarifa (BPS)
                df_clean['Tarifa_BPS_Simulada'] = df_clean['Volumen_USD_Simulado'].apply(calcular_tarifa_bps)
                
                # 3. Calcular Ingreso Estimado (Volumen * BPS / 10000)
                df_clean['Ingreso_USD_Simulado'] = df_clean['Volumen_USD_Simulado'] * (df_clean['Tarifa_BPS_Simulada'] / 10000)
                
                # --- RESULTADOS ---
                st.divider()
                st.subheader("3. Resultados de la Simulación")
                
                # KPIs Generales
                col1, col2, col3 = st.columns(3)
                total_usd = df_clean['Ingreso_USD_Simulado'].sum()
                volumen_total = df_clean['Volumen_USD_Simulado'].sum()
                
                col1.metric("Ingreso Total Proyectado", f"${total_usd:,.0f} USD")
                col2.metric("Volumen Total Procesado", f"${volumen_total/1_000_000:,.0f} MM USD")
                col3.metric("Tarifa Implícita Promedio", f"{(total_usd/volumen_total)*10000:.2f} bps")
                
                # Tabla Resumen por Cliente
                st.subheader("Detalle por Cliente")
                df_agrupado = df_clean.groupby('Corredor')[['Volumen_USD_Simulado', 'Ingreso_USD_Simulado']].sum().sort_values('Ingreso_USD_Simulado', ascending=False)
                st.dataframe(df_agrupado.style.format("${:,.2f}"))
                
                # Gráfico
                st.subheader("Distribución de Ingresos")
                st.bar_chart(df_agrupado['Ingreso_USD_Simulado'].head(10)) # Top 10 clientes
                
                # Opción de descarga
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar Reporte Detallado (CSV)",
                    csv,
                    "reporte_simulacion_rv.csv",
                    "text/csv",
                    key='download-csv'
                )
                
        else:
            st.error("No encontré la hoja llamada 'A.3 BBDD Neg'. Verifica el nombre en tu Excel.")
            st.write("Hojas disponibles detectadas:", xls.sheet_names)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.warning("Consejo: Asegúrate de que el archivo no esté protegido con contraseña.")

else:
    st.info("Esperando archivo Excel...")