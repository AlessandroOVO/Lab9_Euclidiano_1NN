import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('bezdekIris.data', header=None)
X = data.iloc[:, :-1].values  # características
y = data.iloc[:, -1].values    # etiquetas

# Función para calcular la distancia euclidiana
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Función para clasificar una muestra usando 1NN
def classify_1nn(X_train, y_train, sample):
    distances = [euclidean_distance(sample, x) for x in X_train]
    nearest_neighbor_index = np.argmin(distances)
    return y_train[nearest_neighbor_index]

# Función para mostrar y graficar la matriz de confusión
def show_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    print(f"\nMatriz de Confusión - {title}")
    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Validación Hold-Out 70/30
def hold_out_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    predictions = [classify_1nn(X_train, y_train, sample) for sample in X_test]
    show_confusion_matrix(y_test, predictions, "Hold-Out 70/30")

# 10-Fold Cross-Validation
def k_fold_cross_validation(X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    y_true = []
    y_pred = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        predictions = [classify_1nn(X_train, y_train, sample) for sample in X_test]
        y_true.extend(y_test)
        y_pred.extend(predictions)
    
    show_confusion_matrix(y_true, y_pred, "10-Fold Cross-Validation")

# Leave-One-Out Cross-Validation (LOO)
def leave_one_out_validation(X, y):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        prediction = classify_1nn(X_train, y_train, X_test[0])
        y_true.append(y_test[0])
        y_pred.append(prediction)
    
    show_confusion_matrix(y_true, y_pred, "Leave-One-Out (LOO)")

# Ejecutar las validaciones y mostrar las matrices de confusión
hold_out_validation(X, y)
k_fold_cross_validation(X, y, k=10)
leave_one_out_validation(X, y)
