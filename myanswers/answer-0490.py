import pandas as pd
import numpy as np

def construir_coocurrencia(df, transaccion_col, producto_col):
    # Self-merge por la columna de transacción
    merged = df.merge(df, on=transaccion_col, suffixes=("_a", "_b"))

    col_a = f"{producto_col}_a"
    col_b = f"{producto_col}_b"

    # Eliminar pares donde el producto aparece consigo mismo
    merged = merged[merged[col_a] != merged[col_b]]

    # Contar co-ocurrencias
    matriz = pd.crosstab(
        merged[col_a],
        merged[col_b]
    )

    # Obtener todos los productos y ordenarlos alfabéticamente
    productos = sorted(df[producto_col].unique())

    # Asegurar matriz cuadrada con todos los productos
    matriz = matriz.reindex(
        index=productos,
        columns=productos,
        fill_value=0
    )

    # Diagonal en cero
    np.fill_diagonal(matriz.values, 0)

    matriz.index.name = None
    matriz.columns.name = None

    return matriz
