import io
from typing import Dict, List

import pandas as pd
import streamlit as st

from engine import SimulationConfig, apply_tranches, calculate_simulation
from excel_parser import (
    parse_customer_journey,
    parse_end_to_end,
    parse_negotiation_base,
)
from utils import format_currency, format_bps


st.set_page_config(page_title="Simulador Tarifario RV", layout="wide")
st.title("📊 Simulador de Estructura Tarifaria RV")
st.markdown(
    """
    Esta aplicación reemplaza el cálculo manual de Excel.
    **Instrucciones:** Sube tu archivo `.xlsx` completo y ajusta los parámetros en la barra lateral para simular escenarios.
    """
)


# ---------------------------------------------------------------------------
# Barra lateral: parámetros generales
# ---------------------------------------------------------------------------
st.sidebar.header("1. Variables de Simulación")

st.sidebar.subheader("Macroeconomía (Tasas FX)")
default_fx: Dict[str, float] = {
    "COP": 4066.60,
    "CLP": 907.20,
    "PEN": 3.75,
    "USD": 1.0,
}
fx_rates: Dict[str, float] = {}
for currency, value in default_fx.items():
    fx_rates[currency] = st.sidebar.number_input(
        f"Tasa {currency}/USD", value=float(value), step=value * 0.01 if value != 1 else 0.1
    )

st.sidebar.subheader("Estructura de Tarifas (BPS)")
st.sidebar.info("Define los tramos y tarifas a aplicar sobre el Volumen USD.")
limite_tramo_1 = st.sidebar.number_input("Límite Tramo 1 (USD)", value=250_000_000, step=1_000_000)
limite_tramo_2 = st.sidebar.number_input("Límite Tramo 2 (USD)", value=500_000_000, step=1_000_000)

tranche_bps: List[float] = [
    st.sidebar.slider("Tarifa Tramo 1 (bps)", 0.0, 2.0, 0.60),
    st.sidebar.slider("Tarifa Tramo 2 (bps)", 0.0, 2.0, 0.50),
    st.sidebar.slider("Tarifa Tramo 3 (> Límite 2)", 0.0, 2.0, 0.40),
]

tranche_limits = [limite_tramo_1, limite_tramo_2]

config = SimulationConfig(fx_rates=fx_rates, tranche_limits=tranche_limits, tranche_bps=tranche_bps)


# ---------------------------------------------------------------------------
# Carga de archivo y listado de hojas
# ---------------------------------------------------------------------------
st.header("2. Carga de Datos")
uploaded_file = st.file_uploader(
    "Sube tu archivo Excel 'Modelamiento Estructura Tarifaria'", type=["xlsx"], help="Debe incluir las hojas: 1. Parametros, A.3 BBDD Neg, 6. Customer Journey y EndToEnd"
)

if uploaded_file is None:
    st.info("Esperando archivo Excel...")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded_file)
except Exception as exc:  # pragma: no cover - lectura depende de archivo usuario
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

st.success("Archivo cargado correctamente.")
st.caption(f"Hojas detectadas: {', '.join(xls.sheet_names)}")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
with st.spinner("Procesando hojas principales..."):
    customer_journey_df, journey_error = parse_customer_journey(xls)
    end_to_end_df, end_error = parse_end_to_end(xls)
    negotiation_df, neg_error = parse_negotiation_base(xls)


# ---------------------------------------------------------------------------
# Tabs de navegación
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Customer Journey",
    "EndToEnd",
    "Simulación por corredor",
])


# ---------------------------------------------------------------------------
# Sección Customer Journey
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Customer Journey")
    if journey_error:
        st.error(journey_error)
    elif customer_journey_df is None or customer_journey_df.empty:
        st.warning("No se encontraron datos en la hoja '6. Customer Journey'.")
    else:
        total_actual = customer_journey_df["Ingreso Actual"].sum()
        total_prop = customer_journey_df["Ingreso Propuesta"].sum()
        variacion = total_prop - total_actual
        variacion_pct = (variacion / total_actual) * 100 if total_actual else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ingreso Actual (MM USD)", format_currency(total_actual / 1_000_000))
        col2.metric("Ingreso Propuesta (MM USD)", format_currency(total_prop / 1_000_000))
        col3.metric("Variación (MM USD)", format_currency(variacion / 1_000_000))
        col4.metric("Variación %", f"{variacion_pct:.2f}%")

        st.markdown("### Detalle por Producto")
        clean_df = customer_journey_df[customer_journey_df["UN"].str.lower() != "nuam"]
        st.dataframe(clean_df, use_container_width=True)

        st.bar_chart(
            data=clean_df.set_index("Producto")[
                ["Ingreso Actual", "Ingreso Propuesta"]
            ],
            use_container_width=True,
        )

        with st.expander("Ver datos crudos"):
            st.dataframe(customer_journey_df)


# ---------------------------------------------------------------------------
# Sección EndToEnd
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("EndToEnd")
    if end_error:
        st.error(end_error)
    elif end_to_end_df is None or end_to_end_df.empty:
        st.warning("No se encontraron datos en la hoja 'EndToEnd'.")
    else:
        st.dataframe(end_to_end_df, use_container_width=True)

        total_row = end_to_end_df[end_to_end_df["Subbloque"].str.lower() == "total"]
        if not total_row.empty:
            tot = total_row.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Ingreso Colombia", format_currency(tot.get("Colombia_ingresos", 0)))
            col2.metric("Ingreso Perú", format_currency(tot.get("Perú_ingresos", 0)))
            col3.metric("Ingreso Chile", format_currency(tot.get("Chile_ingresos", 0)))

        with st.expander("Ver datos crudos"):
            st.dataframe(end_to_end_df)


# ---------------------------------------------------------------------------
# Sección Simulación por corredor
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Simulación por corredor (A.3 BBDD Neg)")
    if neg_error:
        st.error(neg_error)
    elif negotiation_df is None or negotiation_df.empty:
        st.warning("No se encontraron datos en la hoja 'A.3 BBDD Neg'.")
    else:
        st.dataframe(negotiation_df.head(), use_container_width=True)

        if st.button("🚀 Ejecutar Simulación End-to-End"):
            simulation = calculate_simulation(negotiation_df, config)
            if simulation.empty:
                st.warning("No se pudieron calcular resultados de simulación.")
            else:
                total_income = simulation["Ingreso_USD_Simulado"].sum()
                total_volume = simulation["Volumen_USD_Simulado"].sum()
                implicit_bps = apply_tranches(total_volume, tranche_limits, tranche_bps)

                col1, col2, col3 = st.columns(3)
                col1.metric("Ingreso Total Proyectado", format_currency(total_income))
                col2.metric("Volumen Total Procesado", f"{total_volume/1_000_000:,.0f} MM USD")
                col3.metric("Tarifa Implícita Promedio", format_bps(implicit_bps))

                grouped = simulation.groupby("Corredor")[
                    ["Volumen_USD_Simulado", "Ingreso_USD_Simulado"]
                ].sum().sort_values("Ingreso_USD_Simulado", ascending=False)

                st.markdown("### Detalle por Corredor")
                st.dataframe(grouped, use_container_width=True)

                st.bar_chart(grouped.head(10)["Ingreso_USD_Simulado"], use_container_width=True)

                csv_buffer = io.StringIO()
                simulation.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Descargar Reporte Detallado (CSV)",
                    csv_buffer.getvalue(),
                    "reporte_simulacion_rv.csv",
                    "text/csv",
                    key="download-csv",
                )


st.caption("MVP preparado para migrar cálculos detallados desde Excel a Python.")