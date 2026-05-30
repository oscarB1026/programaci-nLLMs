import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def mean_cluster_distance(X: pd.DataFrame, k: int) -> float:
    """
    Entrena KMeans y calcula la distancia euclidiana promedio 
    de cada punto a su centroide asignado.
    """
    # Configurar el modelo idéntico al del compañero para congelar la aleatoriedad
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    # Obtener etiquetas y coordenadas de los centroides
    labels = model.labels_
    centroids = model.cluster_centers_

    # Calcular distancias de forma eficiente
    distances = []
    for i in range(len(X)):
        punto = X.iloc[i].values
        centroide = centroids[labels[i]]
        dist = np.linalg.norm(punto - centroide)  # Distancia euclidiana
        distances.append(dist)

    return float(np.mean(distances))
