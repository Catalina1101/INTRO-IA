import pandas as pd
import numpy as np
 
def construir_coocurrencia(df, transaccion_col, producto_col):
    """
    Construye una matriz de co-ocurrencia de productos a partir de transacciones.
 
    Parámetros:
        df             (pd.DataFrame): DataFrame con columnas de transacción y producto.
        transaccion_col      (str): Nombre de la columna con el ID de transacción.
        producto_col         (str): Nombre de la columna con el nombre del producto.
 
    Retorna:
        pd.DataFrame: Matriz cuadrada y simétrica con co-ocurrencias. Filas y columnas
                      ordenadas alfabéticamente, diagonal = 0, valores enteros.
    """
    # 1. Self-merge: genera todos los pares (producto_a, producto_b) por transacción
    merged = df.merge(df, on=transaccion_col, suffixes=("_a", "_b"))
 
    col_a = f"{producto_col}_a"
    col_b = f"{producto_col}_b"
 
    # 2. Eliminar pares donde el producto aparece consigo mismo
    merged = merged[merged[col_a] != merged[col_b]]
 
    # 3. Contar transacciones únicas por cada par de productos
    conteo = (
        merged
        .groupby([col_a, col_b])[transaccion_col]
        .nunique()
        .reset_index()
    )
    conteo.columns = ["producto_a", "producto_b", "coocurrencias"]
 
    # 4. Construir tabla pivote (matriz de co-ocurrencia)
    matriz = conteo.pivot(
        index="producto_a",
        columns="producto_b",
        values="coocurrencias"
    ).fillna(0).astype(int)
 
    # 5. Asegurar que todos los productos aparezcan en filas Y columnas
    todos = sorted(df[producto_col].unique())
    matriz = matriz.reindex(index=todos, columns=todos, fill_value=0)
 
    # 6. Diagonal en cero y limpiar nombres de índice/columnas
    vals = matriz.values.copy()
    np.fill_diagonal(vals, 0)
    matriz = pd.DataFrame(vals, index=matriz.index, columns=matriz.columns)
    matriz.index.name = None
    matriz.columns.name = None
 
    return matriz
