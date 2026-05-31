import pandas as pd
import numpy as np
import random
 
def completar_monitoreo_asentamiento(df):
    """
    Completa valores faltantes (0 o NaN) en la columna 'asentamiento'.
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columna 'asentamiento'.
 
    Retorna:
        np.ndarray: Array con valores corregidos.
    """
    valores = df["asentamiento"].values.copy().astype(float)
 
    for i in range(len(valores)):
        if np.isnan(valores[i]) or valores[i] == 0:
 
            # Buscar valor anterior válido
            prev = None
            for j in range(i - 1, -1, -1):
                if not (np.isnan(valores[j]) or valores[j] == 0):
                    prev = valores[j]
                    break
 
            # Buscar valor siguiente válido
            next_ = None
            for j in range(i + 1, len(valores)):
                if not (np.isnan(valores[j]) or valores[j] == 0):
                    next_ = valores[j]
                    break
 
            # Aplicar corrección según disponibilidad
            if prev is not None and next_ is not None:
                valores[i] = (prev + next_) / 2
            elif prev is not None:
                valores[i] = prev
            elif next_ is not None:
                valores[i] = next_
 
    return np.array(valores)
