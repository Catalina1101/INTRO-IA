import numpy as np
from sklearn.decomposition import PCA

def reducir_dimensionalidad_pca(X):
    pca = PCA(n_components=2)
    X_transformada = pca.fit_transform(X)
    
    # Al convertirlo a lista, evitas el ValueError
    return X_transformada.tolist()
