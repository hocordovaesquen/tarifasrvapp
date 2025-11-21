import streamlit as st
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Tarifario RV", layout="wide")

st.title("📊 Simulador de Estructura Tarifaria RV")
st.markdown("Sube tus transacciones y simula diferentes escenarios de tasas y tramos.")

# --- SIDEBAR: CONFIGURACIÓN (LO QUE ERA 1. PARAMETROS) ---
st.sidebar.header("1. Configuración de Escenario")

# Tasas de Cambio Dinámicas
st.sidebar.subheader("Macroeconomía")
tasa_cop = st.sidebar.number_input("Tasa COP/USD", value=4066.60)
tasa_clp = st.sidebar.number_input("Tasa CLP/USD", value=907.20)
tasa_pen = st.sidebar.number_input("Tasa PEN/USD", value=3.75)

tasas = {"COP": tasa_cop, "CLP": tasa_clp, "PEN": tasa_pen, "USD": 1.0}

# Definición de Tramos (Simplificado para el demo)
st.sidebar.subheader("Reglas de Negocio (Tramos)")
bps_tramo_1 = st.sidebar.slider("Tarifa Tramo 1 (< 250MM)", 0.0, 2.0, 0.60)
bps_tramo_2 = st.sidebar.slider("Tarifa Tramo 2 (250-500MM)", 0.0, 2.0, 0.50)
bps_tramo_3 = st.sidebar.slider("Tarifa Tramo 3 (> 500MM)", 0.0, 2.0, 0.40)

# --- LÓGICA DE CÁLCULO ---
def convertir_a_usd(monto, moneda):
    if moneda == "USD": return monto
    return monto / tasas.get(moneda, 1)

def obtener_tarifa(volumen_usd):
    # Lógica simplificada basada en tus sliders
    if volumen_usd <= 250_000_000: return bps_tramo_1
    elif volumen_usd <= 500_000_000: return bps_tramo_2
    else: return bps_tramo_3

# --- ÁREA PRINCIPAL: INPUT DE DATOS ---
st.header("2. Carga de Datos (Transacciones)")
uploaded_file = st.file_uploader("Sube el archivo 'A.3 BBDD Neg' (CSV)", type="csv")

if uploaded_file is not None:
    # Leemos el CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostramos una vista previa
    st.write("Vista previa de los datos cargados:", df.head())

    # Validamos que tenga las columnas necesarias
    # (Asumimos que limpiaste el CSV para tener columnas: 'Cliente', 'Monto Local', 'Moneda')
    # Para este demo, crearemos datos simulados si el CSV no coincide perfectamente,
    # pero aquí iría tu lógica de mapeo.
    
    if st.button("🚀 Ejecutar Simulación"):
        
        resultados = []
        
        # Iteramos sobre las filas del Excel subido
        for index, row in df.iterrows():
            # Adaptar nombres de columnas según tu CSV real
            cliente = row.get('Cliente', 'Desconocido')
            monto_local = row.get('Monto Local', 0)
            moneda = row.get('Moneda', 'COP') # Default a COP si no especifica
            
            vol_usd = convertir_a_usd(monto_local, moneda)
            bps = obtener_tarifa(vol_usd)
            ingreso = vol_usd * (bps / 10000)
            
            resultados.append({
                "Cliente": cliente,
                "Volumen USD": vol_usd,
                "Tarifa Aplicada (bps)": bps,
                "Ingreso Estimado (USD)": ingreso
            })
            
        df_res = pd.DataFrame(resultados)
        
        # --- RESULTADOS VISUALES ---
        st.divider()
        st.subheader("3. Resultados de la Simulación")
        
        # Métricas grandes (KPIs)
        total_ingreso = df_res["Ingreso Estimado (USD)"].sum()
        st.metric(label="Ingreso Total Proyectado", value=f"${total_ingreso:,.2f}")
        
        # Tabla detallada
        st.dataframe(df_res.style.format({"Volumen USD": "${:,.2f}", "Ingreso Estimado (USD)": "${:,.2f}"}))
        
        # Gráfico simple
        st.bar_chart(df_res, x="Cliente", y="Ingreso Estimado (USD)")

else:
    st.info("Esperando archivo CSV para iniciar...")