import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix

# Cargar el dataset Iris
df = pd.read_csv('bezdekIris.data', header=None)
datos = df.iloc[:, :-1].values  # Seleccionar todas las columnas menos la última (características)
clases = df.iloc[:, -1].values  # Seleccionar la última columna (etiquetas de clase)


'''-------------------------------Funciones-------------------------'''
# Función para calcular la distancia euclidiana entre puntos
def distancia_euclidiana(punto_user, punto_array):
    return np.sqrt(np.sum((punto_user - punto_array)**2))

# Función para clasificar usando KNN
def KNN_clasificar(k, punto_usuario, datos_entrenamiento, clases_entrenamiento):
    distancias = np.array([distancia_euclidiana(punto_usuario, punto) for punto in datos_entrenamiento])
    indices_ordenados = np.argsort(distancias)
    clases_cercanas = clases_entrenamiento[indices_ordenados[:k]]
    
    claseA = np.sum(clases_cercanas == 'A')
    claseB = np.sum(clases_cercanas == 'B')
    
    if claseA > claseB:
        return 'A'
    elif claseB > claseA:
        return 'B'
    else:
        return np.random.choice(['A', 'B'])  # Estrategia de desempate aleatoria
    
# Función para calcular precisión
def calcular_precision(clases_reales, clases_predichas):
    correctas = np.sum(clases_reales == clases_predichas)
    return correctas / len(clases_reales)

# Validación Hold-Out 70/30
def validar_hold_out(datos, clases, k):
    datos_entrenamiento, datos_prueba, clases_entrenamiento, clases_prueba = train_test_split(
        datos, clases, test_size=0.3, random_state=42
    )
    clases_predichas = [KNN_clasificar(k, punto, datos_entrenamiento, clases_entrenamiento) for punto in datos_prueba]
    precision = calcular_precision(clases_prueba, clases_predichas)
    print(f"Precisión Hold-Out (70/30): {precision:.2f}")
    
    # Matriz de confusión
    matriz_confusion = confusion_matrix(clases_prueba, clases_predichas, labels=['A', 'B'])
    print("Matriz de Confusión Hold-Out:")
    print(matriz_confusion)

# Validación 10-Fold Cross-Validation
def validar_cross_validation(datos, clases, k, pliegues=10):
    kf = KFold(n_splits=pliegues, shuffle=True, random_state=42)
    clases_reales = []
    clases_predichas = []
    
    for train_index, test_index in kf.split(datos):
        datos_entrenamiento, datos_prueba = datos[train_index], datos[test_index]
        clases_entrenamiento, clases_prueba = clases[train_index], clases[test_index]
        
        predicciones = [KNN_clasificar(k, punto, datos_entrenamiento, clases_entrenamiento) for punto in datos_prueba]
        clases_reales.extend(clases_prueba)
        clases_predichas.extend(predicciones)
    
    precision = calcular_precision(clases_reales, clases_predichas)
    print(f"Precisión promedio 10-Fold Cross-Validation: {precision:.2f}")
    
    # Matriz de confusión
    matriz_confusion = confusion_matrix(clases_reales, clases_predichas, labels=['A', 'B'])
    print("Matriz de Confusión 10-Fold Cross-Validation:")
    print(matriz_confusion)

# Validación Leave-One-Out
def validar_leave_one_out(datos, clases, k):
    loo = LeaveOneOut()
    clases_reales = []
    clases_predichas = []
    
    for train_index, test_index in loo.split(datos):
        datos_entrenamiento, datos_prueba = datos[train_index], datos[test_index]
        clases_entrenamiento, clase_prueba = clases[train_index], clases[test_index[0]]
        
        prediccion = KNN_clasificar(k, datos_prueba[0], datos_entrenamiento, clases_entrenamiento)
        clases_reales.append(clase_prueba)
        clases_predichas.append(prediccion)
    
    precision = calcular_precision(clases_reales, clases_predichas)
    print(f"Precisión promedio Leave-One-Out: {precision:.2f}")
    
    # Matriz de confusión
    matriz_confusion = confusion_matrix(clases_reales, clases_predichas, labels=['A', 'B'])
    print("Matriz de Confusión Leave-One-Out:")
    print(matriz_confusion)

'''--------------Fase de ejecución------------------'''
# Generación de datasets de clases A y B
datasetX1, datasetY1 = np.random.uniform(0, 10, 30), np.random.uniform(0, 10, 30)
datasetX2, datasetY2 = np.random.uniform(0, 10, 20), np.random.uniform(0, 10, 20)

dataset1 = np.column_stack((datasetX1, datasetY1))
dataset2 = np.column_stack((datasetX2, datasetY2))

# Dataset combinado
datasetX = np.concatenate((datasetX1, datasetX2))
datasetY = np.concatenate((datasetY1, datasetY2))
dataset3 = np.column_stack((datasetX, datasetY))

# Asignación de clases
datos = dataset3
clases = np.array(['A'] * len(dataset1) + ['B'] * len(dataset2))

# Definir el número de vecinos
k = int(input('Elija el número de vecinos (impar): '))

# Validación Hold-Out
validar_hold_out(datos, clases, k)

# Validación 10-Fold Cross-Validation
validar_cross_validation(datos, clases, k)

# Validación Leave-One-Out
validar_leave_one_out(datos, clases, k)

# Visualización de una parte del dataset
plt.scatter(datos[clases == 'A'][:, 0], datos[clases == 'A'][:, 1], color='purple', label='Clase A')
plt.scatter(datos[clases == 'B'][:, 0], datos[clases == 'B'][:, 1], color='green', label='Clase B')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()
