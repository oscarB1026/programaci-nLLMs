import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def generar_caso_de_uso_pipeline_svr():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función pipeline_svr.
    """

    # 1. Configuración aleatoria
    n_filas = random.randint(50, 150)
    n_features = random.randint(2, 5)
    test_size = round(random.choice([0.2, 0.25, 0.3]), 2)

    # 2. Generar características aleatorias
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    X_data = np.random.uniform(0, 10, size=(n_filas, n_features))

    # 3. Generar variable objetivo como combinación no lineal
    # para que SVR tenga sentido
    coeficientes = np.random.uniform(1, 5, size=n_features)
    ruido = np.random.normal(0, 0.5, size=n_filas)
    y_data = np.sin(X_data @ coeficientes / n_features) * 10 + ruido

    # 4. Construir el DataFrame
    target_col = 'objetivo'
    df = pd.DataFrame(X_data, columns=feature_cols)
    df[target_col] = y_data

    # ---------------------------------------------------------
    # 5. Construir el INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'test_size': test_size
    }

    # ---------------------------------------------------------
    # 6. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener pipeline_svr
    # ---------------------------------------------------------

    # A. Separar X e y
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    # B. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # C. Construir y entrenar el pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    pipeline.fit(X_train, y_train)

    # D. Predecir y calcular MAE
    y_pred = pipeline.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 4)

    output_data = (y_pred, mae)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    i, o = generar_caso_de_uso_pipeline_svr()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----')
    y_pred, mae = o
    print('\n y_pred :\n', y_pred)
    print('\n mae :\n', mae)
