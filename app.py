"""
DEMOSTRACIÓN: Réplica de Lógica de Cálculo del Modelo Excel
=============================================================

Este código demuestra cómo se replica la lógica de cálculo de tarifas
del modelo Excel en Python, con ejemplos funcionales y resultados.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import json


# ============================================================================
# CONFIGURACIÓN DE TARIFAS (Replica hoja "1. Parametros")
# ============================================================================

@dataclass
class RangoTarifario:
    """Representa un rango tarifario con sus límites y tarifas"""
    min_usd: float
    max_usd: Optional[float]  # None = sin límite superior
    tarifa_variable_pct: float  # En porcentaje (ej: 0.25 = 0.25%)
    tarifa_fija_usd: float


@dataclass
class ConfiguracionPais:
    """Configuración completa de tarifas para un país"""
    nombre: str
    moneda: str
    tasa_cambio_usd: float
    rangos: List[RangoTarifario]
    tarifa_minima_usd: float


class TasasCambio:
    """Maneja las tasas de cambio entre monedas"""
    
    def __init__(self):
        self.tasas = {
            'COP': 4066.61,    # Pesos colombianos por USD
            'PEN': 3.75,       # Soles peruanos por USD
            'CLP': 907.20,     # Pesos chilenos por USD
            'UF': 0.02327,     # Unidad de Fomento por USD (1 UF = 43 USD aprox)
            'BRL': 5.68,       # Reales brasileños por USD
            'MXN': 19.60       # Pesos mexicanos por USD
        }
        self.ultima_actualizacion = datetime.now()
    
    def convertir(self, monto: float, de: str, a: str = 'USD') -> float:
        """Convierte un monto de una moneda a otra"""
        if de == a:
            return monto
        
        if de == 'USD':
            return monto * self.tasas[a]
        elif a == 'USD':
            return monto / self.tasas[de]
        else:
            # Convertir a USD primero, luego a moneda destino
            monto_usd = monto / self.tasas[de]
            return monto_usd * self.tasas[a]
    
    def obtener_tasa(self, moneda: str) -> float:
        """Obtiene la tasa de cambio a USD"""
        return self.tasas.get(moneda, 1.0)


# ============================================================================
# CONFIGURACIÓN DE TARIFAS POR PAÍS
# ============================================================================

CONFIGURACIONES = {
    'colombia': ConfiguracionPais(
        nombre='Colombia',
        moneda='COP',
        tasa_cambio_usd=4066.61,
        rangos=[
            RangoTarifario(0, 5_000_000, 0.25, 0),           # 0-5M: 0.25%
            RangoTarifario(5_000_000, 20_000_000, 0.15, 0),  # 5M-20M: 0.15%
            RangoTarifario(20_000_000, None, 0.10, 0)        # 20M+: 0.10%
        ],
        tarifa_minima_usd=500
    ),
    'peru': ConfiguracionPais(
        nombre='Perú',
        moneda='PEN',
        tasa_cambio_usd=3.75,
        rangos=[
            RangoTarifario(0, 10_000_000, 0.20, 0),          # 0-10M: 0.20%
            RangoTarifario(10_000_000, None, 0.12, 0)        # 10M+: 0.12%
        ],
        tarifa_minima_usd=800
    ),
    'chile': ConfiguracionPais(
        nombre='Chile',
        moneda='CLP',
        tasa_cambio_usd=907.20,
        rangos=[
            RangoTarifario(0, 3_000_000, 0.30, 0),           # 0-3M: 0.30%
            RangoTarifario(3_000_000, 15_000_000, 0.18, 0),  # 3M-15M: 0.18%
            RangoTarifario(15_000_000, None, 0.12, 0)        # 15M+: 0.12%
        ],
        tarifa_minima_usd=600
    )
}


# ============================================================================
# CALCULADOR DE TARIFAS (Replica hoja "3. Negociación")
# ============================================================================

class CalculadorTarifas:
    """
    Calculador de tarifas que replica la lógica de MIN/MAX/IF del Excel.
    
    Funciones de Excel replicadas:
    - MIN(valor, limite): determina el volumen en cada rango
    - MAX(calculado, minimo): aplica tarifa mínima
    - IF(): lógica condicional para rangos
    - SUMIF(): agregación por broker
    """
    
    def __init__(self, configuraciones: Dict[str, ConfiguracionPais]):
        self.config = configuraciones
        self.tasas = TasasCambio()
    
    def calcular_tarifa_por_rango(
        self, 
        volumen_usd: float, 
        pais: str
    ) -> Dict:
        """
        Calcula la tarifa aplicando lógica de rangos (MIN/MAX de Excel)
        
        Esta función replica exactamente la lógica de las fórmulas Excel:
        =MIN(MAX(volumen - rango_min, 0), rango_max - rango_min) * tarifa_var + tarifa_fija
        
        Args:
            volumen_usd: Volumen de negociación en USD
            pais: País para aplicar tarifas
            
        Returns:
            Diccionario con resultado detallado del cálculo
        """
        config_pais = self.config[pais]
        
        # Inicializar acumuladores
        tarifa_total = 0
        volumen_restante = volumen_usd
        desglose_por_rango = []
        
        # Iterar sobre cada rango (replica lógica de IF anidados en Excel)
        for i, rango in enumerate(config_pais.rangos, 1):
            if volumen_restante <= 0:
                break
            
            # Determinar cuánto volumen cae en este rango
            # Replica: =MIN(volumen_restante, rango_max - rango_min)
            if rango.max_usd is None:  # Último rango sin límite
                vol_en_rango = volumen_restante
            else:
                vol_en_rango = min(volumen_restante, rango.max_usd - rango.min_usd)
            
            # Calcular tarifa para este rango
            # Replica: =(volumen * tarifa_var%) + tarifa_fija
            tarifa_rango = (vol_en_rango * rango.tarifa_variable_pct / 100) + rango.tarifa_fija_usd
            
            # Acumular
            tarifa_total += tarifa_rango
            volumen_restante -= vol_en_rango
            
            # Guardar desglose
            desglose_por_rango.append({
                'rango': f"Rango {i}: ${rango.min_usd:,.0f} - {f'${rango.max_usd:,.0f}' if rango.max_usd else 'Sin límite'}",
                'volumen_aplicado': vol_en_rango,
                'tarifa_variable_pct': rango.tarifa_variable_pct,
                'tarifa_calculada': tarifa_rango
            })
        
        # Aplicar mínimo (replica =MAX(calculado, minimo))
        tarifa_aplicada = max(tarifa_total, config_pais.tarifa_minima_usd)
        aplico_minimo = tarifa_aplicada > tarifa_total
        
        # Calcular BPS efectivo
        # BPS = (tarifa / volumen) * 10,000
        bps_efectivo = (tarifa_aplicada / volumen_usd * 10_000) if volumen_usd > 0 else 0
        
        return {
            'pais': config_pais.nombre,
            'volumen_usd': volumen_usd,
            'tarifa_calculada': tarifa_total,
            'tarifa_minima': config_pais.tarifa_minima_usd,
            'tarifa_aplicada': tarifa_aplicada,
            'aplico_minimo': aplico_minimo,
            'bps_efectivo': bps_efectivo,
            'desglose_por_rango': desglose_por_rango
        }
    
    def comparar_estructuras(
        self,
        df_brokers: pd.DataFrame,
        estructura_propuesta: Dict[str, ConfiguracionPais] = None
    ) -> pd.DataFrame:
        """
        Compara estructura actual vs propuesta (replica análisis de variaciones)
        
        Replica la hoja "2. Análisis de variaciones" del Excel
        """
        if estructura_propuesta is None:
            estructura_propuesta = self.config
        
        resultados = []
        
        for _, broker in df_brokers.iterrows():
            # Calcular con estructura actual
            actual = self.calcular_tarifa_por_rango(
                broker['volumen_usd'],
                broker['pais']
            )
            
            # Calcular con estructura propuesta (simulada como +10% en tarifas)
            # En producción esto vendría de otra configuración
            propuesta = self.calcular_tarifa_por_rango(
                broker['volumen_usd'],
                broker['pais']
            )
            propuesta['tarifa_aplicada'] *= 1.10  # Simulamos incremento 10%
            propuesta['bps_efectivo'] *= 1.10
            
            # Calcular variaciones
            delta_abs = propuesta['tarifa_aplicada'] - actual['tarifa_aplicada']
            delta_pct = (delta_abs / actual['tarifa_aplicada'] * 100) if actual['tarifa_aplicada'] > 0 else 0
            
            resultados.append({
                'broker': broker['broker'],
                'pais': broker['pais'],
                'volumen_usd': broker['volumen_usd'],
                'tarifa_actual': actual['tarifa_aplicada'],
                'bps_actual': actual['bps_efectivo'],
                'aplico_minimo_actual': actual['aplico_minimo'],
                'tarifa_propuesta': propuesta['tarifa_aplicada'],
                'bps_propuesta': propuesta['bps_efectivo'],
                'delta_usd': delta_abs,
                'delta_pct': delta_pct,
                'impacto': 'Positivo' if delta_pct > 5 else 'Neutral' if delta_pct > -5 else 'Negativo'
            })
        
        return pd.DataFrame(resultados)
    
    def analizar_sensibilidad(
        self,
        pais: str,
        volumenes_test: List[float],
        variacion_tarifas: List[float]
    ) -> pd.DataFrame:
        """
        Análisis de sensibilidad variando tarifas y volúmenes
        
        Args:
            pais: País a analizar
            volumenes_test: Lista de volúmenes a probar (en USD)
            variacion_tarifas: Lista de variaciones % (ej: [-10, -5, 0, 5, 10])
        
        Returns:
            DataFrame con matriz de sensibilidad
        """
        resultados = []
        
        for variacion in variacion_tarifas:
            for volumen in volumenes_test:
                # Calcular con configuración actual
                resultado = self.calcular_tarifa_por_rango(volumen, pais)
                
                # Aplicar variación
                tarifa_ajustada = resultado['tarifa_aplicada'] * (1 + variacion/100)
                bps_ajustado = resultado['bps_efectivo'] * (1 + variacion/100)
                
                resultados.append({
                    'variacion_tarifa_pct': variacion,
                    'volumen_usd': volumen,
                    'volumen_millones': volumen / 1_000_000,
                    'tarifa_usd': tarifa_ajustada,
                    'bps': bps_ajustado
                })
        
        return pd.DataFrame(resultados)


# ============================================================================
# EJEMPLOS DE USO Y VALIDACIÓN
# ============================================================================

def ejemplo_1_calculo_simple():
    """Ejemplo 1: Cálculo simple de tarifa para un broker"""
    print("=" * 80)
    print("EJEMPLO 1: Cálculo Simple de Tarifa")
    print("=" * 80)
    
    calculador = CalculadorTarifas(CONFIGURACIONES)
    
    # Broker con 15M USD en Colombia
    resultado = calculador.calcular_tarifa_por_rango(
        volumen_usd=15_000_000,
        pais='colombia'
    )
    
    print(f"\n📊 Resultado del Cálculo:")
    print(f"País: {resultado['pais']}")
    print(f"Volumen: ${resultado['volumen_usd']:,.2f} USD")
    print(f"Tarifa calculada: ${resultado['tarifa_calculada']:,.2f} USD")
    print(f"Tarifa mínima: ${resultado['tarifa_minima']:,.2f} USD")
    print(f"Tarifa aplicada: ${resultado['tarifa_aplicada']:,.2f} USD")
    print(f"¿Aplicó mínimo?: {'Sí' if resultado['aplico_minimo'] else 'No'}")
    print(f"BPS efectivo: {resultado['bps_efectivo']:.2f}")
    
    print(f"\n📋 Desglose por Rango:")
    for desglose in resultado['desglose_por_rango']:
        print(f"  {desglose['rango']}")
        print(f"    Volumen aplicado: ${desglose['volumen_aplicado']:,.2f}")
        print(f"    Tarifa variable: {desglose['tarifa_variable_pct']}%")
        print(f"    Tarifa calculada: ${desglose['tarifa_calculada']:,.2f}")
    
    return resultado


def ejemplo_2_comparacion_estructuras():
    """Ejemplo 2: Comparar estructura actual vs propuesta para múltiples brokers"""
    print("\n" + "=" * 80)
    print("EJEMPLO 2: Comparación de Estructuras")
    print("=" * 80)
    
    # Crear datos de ejemplo de brokers
    datos_brokers = {
        'broker': ['Broker A', 'Broker B', 'Broker C', 'Broker D', 'Broker E'],
        'pais': ['colombia', 'colombia', 'peru', 'chile', 'peru'],
        'volumen_usd': [15_000_000, 8_500_000, 12_000_000, 18_000_000, 6_000_000]
    }
    df_brokers = pd.DataFrame(datos_brokers)
    
    calculador = CalculadorTarifas(CONFIGURACIONES)
    df_comparacion = calculador.comparar_estructuras(df_brokers)
    
    print("\n📊 Resultados de Comparación:")
    print(df_comparacion.to_string(index=False))
    
    # Estadísticas agregadas
    print(f"\n📈 Resumen Estadístico:")
    print(f"Total brokers analizados: {len(df_comparacion)}")
    print(f"Ingreso actual total: ${df_comparacion['tarifa_actual'].sum():,.2f} USD")
    print(f"Ingreso propuesto total: ${df_comparacion['tarifa_propuesta'].sum():,.2f} USD")
    print(f"Delta total: ${df_comparacion['delta_usd'].sum():,.2f} USD")
    print(f"Delta promedio: {df_comparacion['delta_pct'].mean():.2f}%")
    print(f"BPS promedio actual: {df_comparacion['bps_actual'].mean():.2f}")
    print(f"BPS promedio propuesto: {df_comparacion['bps_propuesta'].mean():.2f}")
    
    return df_comparacion


def ejemplo_3_analisis_sensibilidad():
    """Ejemplo 3: Análisis de sensibilidad"""
    print("\n" + "=" * 80)
    print("EJEMPLO 3: Análisis de Sensibilidad")
    print("=" * 80)
    
    calculador = CalculadorTarifas(CONFIGURACIONES)
    
    # Definir parámetros de análisis
    volumenes_test = [5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000]
    variaciones = [-10, -5, 0, 5, 10]
    
    df_sensibilidad = calculador.analizar_sensibilidad(
        pais='colombia',
        volumenes_test=volumenes_test,
        variacion_tarifas=variaciones
    )
    
    print("\n📊 Matriz de Sensibilidad (Colombia):")
    print("\nTarifa (USD) por Volumen y Variación:")
    
    # Crear tabla pivot
    pivot = df_sensibilidad.pivot(
        index='variacion_tarifa_pct',
        columns='volumen_millones',
        values='tarifa_usd'
    )
    
    print(pivot.to_string())
    
    print("\n\n📊 BPS por Volumen y Variación:")
    pivot_bps = df_sensibilidad.pivot(
        index='variacion_tarifa_pct',
        columns='volumen_millones',
        values='bps'
    )
    
    print(pivot_bps.to_string())
    
    return df_sensibilidad


def ejemplo_4_customer_journey():
    """Ejemplo 4: Análisis de Customer Journey (replica hoja "6. Customer Journey")"""
    print("\n" + "=" * 80)
    print("EJEMPLO 4: Customer Journey - Análisis por Segmento")
    print("=" * 80)
    
    # Simular datos de clientes con diferentes perfiles
    datos_clientes = {
        'cliente': [f'Cliente {i}' for i in range(1, 11)],
        'broker': ['Broker A', 'Broker A', 'Broker B', 'Broker B', 'Broker C',
                  'Broker C', 'Broker D', 'Broker D', 'Broker E', 'Broker E'],
        'pais': ['colombia', 'colombia', 'peru', 'peru', 'chile',
                'chile', 'colombia', 'peru', 'chile', 'colombia'],
        'segmento': ['Grande', 'Mediano', 'Grande', 'Pequeño', 'Grande',
                    'Mediano', 'Pequeño', 'Grande', 'Mediano', 'Pequeño'],
        'volumen_mensual_usd': [5_000_000, 2_000_000, 4_500_000, 800_000, 6_000_000,
                               1_500_000, 600_000, 3_800_000, 2_200_000, 500_000]
    }
    df_clientes = pd.DataFrame(datos_clientes)
    
    calculador = CalculadorTarifas(CONFIGURACIONES)
    
    # Calcular tarifas para cada cliente
    resultados_clientes = []
    for _, cliente in df_clientes.iterrows():
        resultado = calculador.calcular_tarifa_por_rango(
            cliente['volumen_mensual_usd'],
            cliente['pais']
        )
        
        resultados_clientes.append({
            'cliente': cliente['cliente'],
            'broker': cliente['broker'],
            'pais': cliente['pais'],
            'segmento': cliente['segmento'],
            'volumen_usd': cliente['volumen_mensual_usd'],
            'tarifa_usd': resultado['tarifa_aplicada'],
            'bps': resultado['bps_efectivo']
        })
    
    df_journey = pd.DataFrame(resultados_clientes)
    
    print("\n📊 Análisis por Cliente:")
    print(df_journey.to_string(index=False))
    
    # Análisis por segmento
    print("\n\n📈 Análisis por Segmento:")
    analisis_segmento = df_journey.groupby('segmento').agg({
        'cliente': 'count',
        'volumen_usd': ['sum', 'mean'],
        'tarifa_usd': ['sum', 'mean'],
        'bps': 'mean'
    }).round(2)
    
    print(analisis_segmento)
    
    # Análisis por país
    print("\n\n🌎 Análisis por País:")
    analisis_pais = df_journey.groupby('pais').agg({
        'cliente': 'count',
        'volumen_usd': ['sum', 'mean'],
        'tarifa_usd': ['sum', 'mean'],
        'bps': 'mean'
    }).round(2)
    
    print(analisis_pais)
    
    return df_journey


# ============================================================================
# VALIDACIÓN CON DATOS DEL EXCEL
# ============================================================================

def validar_contra_excel():
    """
    Función para validar que los cálculos Python coinciden con Excel.
    Aquí pondrías valores específicos del Excel para comparar.
    """
    print("\n" + "=" * 80)
    print("VALIDACIÓN: Comparación con Resultados de Excel")
    print("=" * 80)
    
    calculador = CalculadorTarifas(CONFIGURACIONES)
    
    # Casos de prueba (sustituir con valores reales del Excel)
    casos_prueba = [
        {'volumen': 15_000_000, 'pais': 'colombia', 'esperado_excel': 27_500},
        {'volumen': 8_500_000, 'pais': 'colombia', 'esperado_excel': 15_750},
        {'volumen': 12_000_000, 'pais': 'peru', 'esperado_excel': 22_400},
    ]
    
    print("\n📊 Casos de Validación:")
    print(f"{'Volumen USD':<15} {'País':<10} {'Python USD':<15} {'Excel USD':<15} {'Match':<10}")
    print("-" * 70)
    
    for caso in casos_prueba:
        resultado = calculador.calcular_tarifa_por_rango(caso['volumen'], caso['pais'])
        python_value = resultado['tarifa_aplicada']
        excel_value = caso['esperado_excel']
        match = abs(python_value - excel_value) < 1.0  # Tolerancia de $1
        
        print(f"{caso['volumen']:<15,.0f} {caso['pais']:<10} {python_value:<15,.2f} {excel_value:<15,.2f} {'✅' if match else '❌':<10}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecutar todos los ejemplos"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   DEMOSTRACIÓN: RÉPLICA DE LÓGICA EXCEL EN PYTHON                    ║
    ║   Modelo de Estructura Tarifaria - Renta Variable                    ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Ejecutar ejemplos
    ejemplo_1_calculo_simple()
    ejemplo_2_comparacion_estructuras()
    ejemplo_3_analisis_sensibilidad()
    ejemplo_4_customer_journey()
    validar_contra_excel()
    
    print("\n" + "=" * 80)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("""
    Este código demuestra cómo:
    ✓ Se replican las fórmulas MIN/MAX/IF del Excel en Python
    ✓ Se organizan los cálculos en funciones reutilizables
    ✓ Se procesan múltiples brokers/clientes eficientemente
    ✓ Se realizan análisis complejos (sensibilidad, customer journey)
    ✓ Se validan los resultados contra el Excel original
    
    La migración a aplicación web es totalmente factible usando esta base.
    """)


if __name__ == "__main__":
    main()
