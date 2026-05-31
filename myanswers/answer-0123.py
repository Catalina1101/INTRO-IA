import numpy as np
from sklearn.decomposition import PCA

def reducir_dimensionalidad_pca(X):
    """
    Reduce la dimensionalidad de X a 2 componentes principales usando PCA.
    
    Parámetros:
        X (np.ndarray): Matriz de datos de forma (n_muestras, n_features)
    
    Retorna:
        np.ndarray: Matriz transformada de forma (n_muestras, 2)
    """
    pca = PCA(n_components=2)
    X_transformada = pca.fit_transform(X)
    return X_transformada
