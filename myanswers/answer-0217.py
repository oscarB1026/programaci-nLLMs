import pandas as pd
import numpy as np

def crear_reporte_ventas_pivot(df):
    """
    Crea una tabla dinámica agrupando las ventas totales por sucursal
    y categoría de producto según las especificaciones del generador.
    """
    # Creamos la tabla pivote siguiendo exactamente los mismos parámetros
    df_output = pd.pivot_table(
        df, 
        values='ventas_totales', 
        index='sucursal', 
        columns='categoria_producto', 
        aggfunc='sum',     # Suma de ventas totales
        fill_value=0       # Rellena combinaciones sin ventas con 0
    )
    
    return df_output
