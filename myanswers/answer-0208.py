import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def mean_cluster_distance(X: pd.DataFrame, k: int) -> float:
    """
    Entrena un modelo KMeans y calcula la distancia euclidiana promedio
    de cada punto a su centroide asignado siguiendo la configuración del caso de uso.
    """
    # 1. Configurar y entrenar el modelo 
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    # 2. Obtener las etiquetas de asignación y las coordenadas de los centroides
    labels = model.labels_
    centroids = model.cluster_centers_

    # Extracción de la matriz de datos para optimizar la velocidad en el bucle
    X_values = X.values
    distances = []

    # 3. Calcular la distancia euclidiana punto por punto
    for i in range(len(X_values)):
        punto = X_values[i]
        centroide = centroids[labels[i]]
        dist = np.linalg.norm(punto - centroide)
        distances.append(dist)

    # 4. Devolver el promedio estricto como float nativo de Python
    return float(np.mean(distances))
