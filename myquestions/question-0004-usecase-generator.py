import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_reduce_genomic_dimensions():
    """
    Genera un caso de prueba para reduce_genomic_dimensions.
    Simula datos de expresión génica con alta correlación latente.
    """
    n_samples = 30
    n_genes = 100
    variance_target = random.uniform(0.80, 0.90)

    # 1. Generar datos con estructura de varianza (no ruido puro)
    latent_space = np.random.randn(n_samples, 5)
    projection = np.random.randn(5, n_genes)
    X = np.dot(latent_space, projection) + np.random.normal(0, 0.1, (n_samples, n_genes))

    input_data = {
        'X': X,
        'variance_target': variance_target
    }

    # 2. Calcular OUTPUT (Ground Truth)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Primero ajustamos un PCA completo para ver la varianza acumulada
    pca_full = PCA(svd_solver='full')
    pca_full.fit(X_scaled)

    evr_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(evr_cum >= variance_target) + 1

    # Ajustar el PCA final con el número de componentes óptimo
    pca_final = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca_final.fit_transform(X_scaled)

    output_data = (pca_final, X_reduced)

    return input_data, output_data

# COMPROBAR FUNCIÓN GENERADORA

print("=" * 50)
print("=== Test 4: generar_caso_de_uso_reduce_genomic_dimensions ===")
try:
    input_data, output_data = generar_caso_de_uso_reduce_genomic_dimensions()

    # Inspeccionar INPUT
    print(f"> INPUT (keys): {list(input_data.keys())}")

    # Inspeccionar OUTPUT
    output_type = type(output_data)
    print(f"> OUTPUT (type): {output_type}")

    if output_type is tuple:
        for idx, item in enumerate(output_data):
            item_type = type(item)
            print(f"  - Element {idx} type: {item_type}")

            # Si es la matriz transformada
            if hasattr(item, 'shape'):
                print(f"  - Element {idx} shape: {item.shape}")
            # Si es el objeto PCA
            elif hasattr(item, 'components_'):
                print(f"  - Element {idx} (PCA) ajustado con {item.n_components_} componentes.")
    else:
         print("  - Advertencia: Se esperaba una tupla.")
except Exception as e:
    print(f"ERROR durante la ejecución: {e}")
print("=" * 50)
