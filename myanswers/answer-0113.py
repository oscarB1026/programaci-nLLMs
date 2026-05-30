import pandas as pd
import numpy as np

def analisis_rendimiento_empleados(df):
    """
    Limpia el reporte de empleados y calcula la eficiencia promedio
    por departamento siguiendo el orden exacto del ground truth del compañero.
    """
    df_clean = df.copy()
    
    # 1. Eliminar las filas donde la columna 'departamento' tenga valores nulos
    df_clean = df_clean.dropna(subset=['departamento'])
    
    # 2. Calcular la mediana de las horas trabajadas tras el filtro e imputar nulos
    mediana_horas = df_clean['horas_trabajadas'].median()
    df_clean['horas_trabajadas'] = df_clean['horas_trabajadas'].fillna(mediana_horas)
    
    # 3. Crear la columna de eficiencia
    df_clean['eficiencia'] = df_clean['proyectos_completados'] / df_clean['horas_trabajadas']
    
    # 4. Agrupar por departamento, calcular promedio y ordenar de mayor a menor
    output = df_clean.groupby('departamento')[['eficiencia']].mean().sort_values(by='eficiencia', ascending=False)
    
    return output
