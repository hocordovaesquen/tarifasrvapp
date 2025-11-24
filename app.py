def calcular_escenario_acceso(df_base: pd.DataFrame, params_por_pais: dict):
    """
    Simulación basada en TRAMOS MENSUALES.
    Lógica:
    1. Agrupar por Pais, Broker y MES.
    2. Sumar volumen mensual.
    3. Aplicar tarifa según en qué tramo cae ese volumen mensual.
    4. Re-agrupar anualmente para el reporte final.
    """
    df = df_base.copy()
    
    if df.empty:
        return df, pd.DataFrame()

    # --- 1. Mapeo de Columnas ---
    col_pais = find_column(df, ["Pais", "País"])
    col_broker = find_column(df, ["Corredor", "Broker", "Cliente"])
    col_monto = find_column(df, ["Monto USD", "Monto_USD", "Volumen USD"])
    col_mes = find_column(df, ["Mes", "Month"]) # Necesitamos la columna Mes del Excel
    col_acceso_actual = find_column(df, ["Cobro Acceso", "Acceso actual", "Ingreso Actual"])

    # Renombrar
    df.rename(columns={
        col_pais: "Pais",
        col_broker: "Broker",
        col_monto: "Monto_USD",
        col_mes: "Mes",
        col_acceso_actual: "Acceso_Actual"
    }, inplace=True)

    # Limpieza
    df["Monto_USD"] = pd.to_numeric(df["Monto_USD"], errors='coerce').fillna(0.0)
    df["Acceso_Actual"] = pd.to_numeric(df["Acceso_Actual"], errors='coerce').fillna(0.0)
    
    # Normalizar Países
    df["Pais"] = df["Pais"].astype(str).str.upper().str.strip()
    df["Pais"] = df["Pais"].replace({
        "PERU": "Perú", "PERÚ": "Perú", 
        "CHILE": "Chile", "COLOMBIA": "Colombia"
    })
    
    # Filtrar solo nuam
    df = df[df["Pais"].isin(["Chile", "Colombia", "Perú"])]

    # --- 2. AGRUPACIÓN MENSUAL (El Corazón del Cambio) ---
    # Sumamos el volumen de cada broker por cada mes
    df_mensual = df.groupby(["Pais", "Broker", "Mes"]).agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum" # Mantenemos el real para comparar luego
    }).reset_index()

    # --- 3. APLICACIÓN DE TARIFAS POR TRAMO ---
    def aplicar_tarifa(row):
        pais = row["Pais"]
        volumen = row["Monto_USD"]
        
        # Recuperar configuración del país
        cfg = params_por_pais.get(pais)
        if not cfg: return 0.0 # Si no hay config, 0

        # Lógica de Tramos
        if volumen < cfg["limite_bajo"]:
            return cfg["tarifa_bajo"]
        elif volumen < cfg["limite_alto"]:
            return cfg["tarifa_medio"]
        else:
            return cfg["tarifa_alto"]

    # Calculamos la cuota mes a mes
    df_mensual["Acceso_Proyectado_Mes"] = df_mensual.apply(aplicar_tarifa, axis=1)

    # --- 4. CONSOLIDACIÓN ANUAL (Para el reporte) ---
    # Ahora volvemos a agrupar por Broker (sumando todos sus meses)
    df_final = df_mensual.groupby(["Pais", "Broker"]).agg({
        "Monto_USD": "sum",             # Volumen total año
        "Acceso_Actual": "sum",         # Lo que pagaron realmente en el año
        "Acceso_Proyectado_Mes": "sum"  # La suma de las 12 cuotas simuladas
    }).reset_index()

    df_final.rename(columns={"Acceso_Proyectado_Mes": "Acceso_Proyectado"}, inplace=True)

    # --- KPIs Finales ---
    df_final["Var_Abs"] = df_final["Acceso_Proyectado"] - df_final["Acceso_Actual"]
    df_final["Var_%"] = np.where(
        df_final["Acceso_Actual"] > 10, 
        (df_final["Var_Abs"] / df_final["Acceso_Actual"]) * 100,
        0.0 # Si antes pagaban 0 (Perú), esto será 0% o infinito. Mejor mostrar absolutos en la tabla.
    )

    # Resumen País
    agg_pais = df_final.groupby("Pais").agg({
        "Monto_USD": "sum",
        "Acceso_Actual": "sum",
        "Acceso_Proyectado": "sum"
    }).reset_index()

    # BPS (Ingreso / Volumen * 10000)
    agg_pais["BPS_Proyectado"] = (agg_pais["Acceso_Proyectado"] / agg_pais["Monto_USD"]) * 10000
    agg_pais["Var_%"] = ((agg_pais["Acceso_Proyectado"] - agg_pais["Acceso_Actual"]) / agg_pais["Acceso_Actual"]) * 100
    
    return df_final, agg_pais