import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
 
def ajustar_pls_quimiometria(X, y, n_componentes):
    """
    Ajusta un modelo PLS para predecir concentraciones a partir de espectros.
 
    Parámetros:
        X             (np.ndarray): Matriz de espectros (n_muestras, n_longitudes_onda).
        y             (np.ndarray): Vector de concentraciones reales (n_muestras,).
        n_componentes       (int): Número de componentes latentes PLS.
 
    Retorna:
        dict con claves:
            'modelo' : PLSRegression entrenado
            'r2'     : Coeficiente de determinación R²
            'mse'    : Error Cuadrático Medio (MSE)
    """
    modelo = PLSRegression(n_components=n_componentes)
    modelo.fit(X, y)
 
    y_pred = modelo.predict(X)
 
    r2  = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
 
    return {
        "modelo": modelo,
        "r2":     r2,
        "mse":    mse,
    }
