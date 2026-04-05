import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_perfil_estadistico():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_perfil_estadistico.
    """

    # 1. Configuración aleatoria
    n_filas = random.randint(30, 80)

    departamentos_posibles = ['Ventas', 'Marketing', 'Finanzas', 'Tecnología',
                              'Recursos Humanos', 'Operaciones', 'Legal']
    n_grupos = random.randint(3, 5)
    departamentos = random.sample(departamentos_posibles, n_grupos)

    grupo_col = 'departamento'
    valor_col = 'salario'

    # 2. Generar el DataFrame aleatorio
    df = pd.DataFrame({
        grupo_col: [random.choice(departamentos) for _ in range(n_filas)],
        valor_col: np.round(np.random.uniform(1500, 8000, size=n_filas), 2)
    })

    # ---------------------------------------------------------
    # 3. Construir el INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'grupo_col': grupo_col,
        'valor_col': valor_col
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener calcular_perfil_estadistico
    # ---------------------------------------------------------

    # A. Agrupar y calcular estadísticas
    resumen = df.groupby(grupo_col)[valor_col].agg(
        media='mean',
        mediana='median',
        desviacion_std='std',
        minimo='min',
        maximo='max'
    )

    # B. Redondear columnas correspondientes
    resumen['media'] = resumen['media'].round(2)
    resumen['mediana'] = resumen['mediana'].round(2)
    resumen['desviacion_std'] = resumen['desviacion_std'].round(2)

    # C. Ordenar alfabéticamente por nombre del grupo
    resumen = resumen.sort_index()

    output_data = resumen

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    i, o = generar_caso_de_uso_calcular_perfil_estadistico()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)
