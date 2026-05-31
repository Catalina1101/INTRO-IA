from sklearn.decomposition import PCA

def reducir_dimensionalidad_pca(X):
    """
    Reduce una matriz de datos a 2 componentes principales usando PCA.
    
    Parámetros
    X : array-like
        Matriz de datos de entrada.

    Retorna
    np.ndarray
        Matriz transformada con 2 dimensiones.
    """
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)
    return X_reducido
