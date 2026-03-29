import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def generar_caso_de_uso_prepare_survival_data():
    """
    Genera un caso de prueba para la función prepare_survival_data.
    Maneja imputación por mediana y escalado robusto.
    """
    n_samples = random.randint(50, 100)

    # 1. Generar datos crudos
    data = {
        'age': np.random.uniform(40, 90, n_samples),
        'bmi': np.random.normal(27, 4, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'event_occurred': np.random.randint(0, 2, n_samples),
        'follow_up_days': np.random.randint(10, 2000, n_samples)
    }

    df = pd.DataFrame(data)

    # 2. Inyectar complejidad: NaNs y Outliers
    df.loc[df.sample(frac=0.1).index, 'age'] = np.nan
    df.loc[0, 'bmi'] = 250.0 # Outlier extremo de IMC

    input_data = {
        'df': df.copy(),
        'event_col': 'event_occurred',
        'duration_col': 'follow_up_days'
    }

    # 3. Calcular OUTPUT (Ground Truth)
    df_expected = df.copy()
    df_expected['event_occurred'] = df_expected['event_occurred'].astype(bool)

    features = ['age', 'bmi', 'blood_pressure']

    # Imputar y escalar solo las columnas de características
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()

    df_expected[features] = imputer.fit_transform(df_expected[features])
    df_expected[features] = scaler.fit_transform(df_expected[features])

    output_data = df_expected

    return input_data, output_data


# COMPORBAR FUNCIÓN GENERADORA

print("=" * 50)
print("=== Test 2: generar_caso_de_uso_prepare_survival_data ===")
try:
    input_data, output_data = generar_caso_de_uso_prepare_survival_data()

    # Inspeccionar INPUT
    print(f"> INPUT (keys): {list(input_data.keys())}")

    # Inspeccionar OUTPUT
    output_type = type(output_data)
    print(f"> OUTPUT (type): {output_type}")

    if issubclass(output_type, pd.DataFrame):
        print(f"  - DataFrame shape: {output_data.shape}")
        print(f"  - Columns: {list(output_data.columns)}")
    else:
        print("  - Advertencia: Se esperaba un DataFrame de Pandas.")
except Exception as e:
    print(f"ERROR durante la ejecución: {e}")
print("=" * 50)
