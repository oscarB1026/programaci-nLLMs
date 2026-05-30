import pandas as pd
import numpy as np

def limpiar_y_resumir_ventas(df):
    """
    Limpia y transforma un DataFrame de ventas siguiendo las especificaciones
    y el orden exacto del generador de casos de uso del compañero.
    """
    # Hacer una copia para evitar modificar el DataFrame original por referencia
    result = df.copy()

    # 1. Eliminar duplicados basados en fecha, producto y precio
    result = result.drop_duplicates(subset=['fecha', 'producto', 'precio'])

    # 2. Eliminar filas donde producto o fecha sean nulos
    result = result.dropna(subset=['producto', 'fecha'])

    # 3. Rellenar precio con la mediana del precio por categoria
    result['precio'] = result.groupby('categoria')['precio'].transform(
        lambda x: x.fillna(x.median())
    )

    # 4. Rellenar cantidad con 1 y descuento con 0.0
    result['cantidad'] = result['cantidad'].fillna(1)
    result['descuento'] = result['descuento'].fillna(0.0)

    # 5. Crear columna total_venta calculada como: precio * cantidad * (1 - descuento)
    result['total_venta'] = result['precio'] * result['cantidad'] * (1 - result['descuento'])

    # 6. Convertir la columna fecha al tipo datetime
    result['fecha'] = pd.to_datetime(result['fecha'])

    # 7. Ordenar por fecha de forma ascendente y reiniciar el índice
    result = result.sort_values('fecha').reset_index(drop=True)

    return result
