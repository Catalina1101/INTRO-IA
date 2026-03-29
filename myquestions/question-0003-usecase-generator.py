import numpy as np
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def generar_caso_de_uso_train_sepsis_detector():
    """
    Genera un caso de prueba para train_sepsis_detector.
    Dataset desbalanceado con optimización de recall.
    """
    n_samples = 250
    n_features = 6

    # 1. Generar X e y (desbalanceado 90/10)
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    class_weight_ratio = random.randint(8, 12)

    input_data = {
        'X': X,
        'y': y,
        'class_weight_ratio': class_weight_ratio
    }

    # 2. Calcular OUTPUT (Ground Truth)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        class_weight={0: 1, 1: class_weight_ratio},
        random_state=42
    )

    # Búsqueda de hiperparámetros reducida para el test
    param_dist = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        scoring='recall', n_iter=2, random_state=42
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    output_data = (best_model, cm)

    return input_data, output_data

# COMPROBAR FUNCIÓN GENERADORA

print("=" * 50)
print("=== Test 3: generar_caso_de_uso_train_sepsis_detector ===")
try:
    input_data, output_data = generar_caso_de_uso_train_sepsis_detector()

    # Inspeccionar INPUT
    print(f"> INPUT (keys): {list(input_data.keys())}")

    # Inspeccionar OUTPUT
    output_type = type(output_data)
    print(f"> OUTPUT (type): {output_type}")

    if output_type is tuple:
        for idx, item in enumerate(output_data):
            item_type = type(item)
            print(f"  - Element {idx} type: {item_type}")

            # Si es un array/matriz
            if hasattr(item, 'shape'):
                print(f"  - Element {idx} shape: {item.shape}")
            # Si es un modelo clasificador
            elif hasattr(item, 'classes_'):
                print(f"  - Element {idx} es un modelo ajustado con clases: {item.classes_}")
    else:
         print("  - Advertencia: Se esperaba una tupla.")
except Exception as e:
    print(f" ERROR durante la ejecución: {e}")
print("=" * 50)
