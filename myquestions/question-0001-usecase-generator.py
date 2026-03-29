import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_extract_hrv_metrics():
    """
    Genera un caso de prueba para la función extract_hrv_metrics.
    Incluye filtrado de artefactos y cálculo de RMSSD/SDNN.
    """
    # 1. Configuración aleatoria
    n_rows = random.randint(60, 120)
    threshold_ms = random.choice([1800, 2000, 2200])

    # 2. Generar datos: Ritmo base con variabilidad + Outliers
    rr_values = np.random.normal(loc=850, scale=120, size=n_rows)
    # Insertar ruido (valores que deben ser filtrados)
    n_noise = int(n_rows * 0.1)
    noise_indices = np.random.choice(n_rows, n_noise, replace=False)
    rr_values[noise_indices[:n_noise//2]] = 150  # Ruido muy bajo
    rr_values[noise_indices[n_noise//2:]] = 3000 # Ruido muy alto

    df = pd.DataFrame({'RR_ms': rr_values})

    # 3. Construir INPUT
    input_data = {
        'rr_intervals_df': df.copy(),
        'threshold_ms': threshold_ms
    }

    # 4. Calcular OUTPUT (Ground Truth)
    # Filtrado según la consigna: [300, threshold_ms]
    df_filtered = df[(df['RR_ms'] >= 300) & (df['RR_ms'] <= threshold_ms)].copy()

    # RMSSD: Raíz de la media de las diferencias al cuadrado
    diffs = df_filtered['RR_ms'].diff().dropna()
    rmssd = np.sqrt(np.mean(diffs**2))

    # SDNN: Desviación estándar de los intervalos válidos
    sdnn = np.std(df_filtered['RR_ms'], ddof=0)

    output_data = {
        "RMSSD": float(rmssd),
        "SDNN": float(sdnn),
        "valid_samples_count": int(len(df_filtered))
    }

    return input_data, output_data

# COMPROBAR FUNCIÓN GENERADORA

print("=" * 50)
print("=== Test 1: generar_caso_de_uso_extract_hrv_metrics ===")
try:
    input_data, output_data = generar_caso_de_uso_extract_hrv_metrics()

    # Inspeccionar INPUT
    print(f"> INPUT (keys): {list(input_data.keys())}")

    # Inspeccionar OUTPUT
    output_type = type(output_data)
    print(f"> OUTPUT (type): {output_type}")

    if output_type is dict:
        print(f"  - Dictionary keys: {list(output_data.keys())}")
        # Mostrar los valores generados para mayor claridad
        for k, v in output_data.items():
            print(f"  - {k}: {v}")
    else:
        print("  - Advertencia: Se esperaba un diccionario.")
except Exception as e:
    print(f"ERROR durante la ejecución: {e}")
print("=" * 50)

