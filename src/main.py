# src/main.py
# --------------------------------------------------------
# Random Forest en Regresión y Clasificación con Scikit-Learn
# --------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import os

# Crear carpeta results si no existe
if not os.path.exists("results"):
    os.makedirs("results")

# ================================
# Parte 1: Regresión con Random Forest
# ================================

print("=== PARTE 1: REGRESIÓN ===")

# Paso 1: Preparación de los Datos
df = pd.read_csv("data/wine-quality.csv")

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Variables predictoras y objetivo
X = df.drop("quality", axis=1)
y_reg = df["quality"]

# División train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Paso 2: Construcción del Modelo
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Paso 3: Evaluación
y_pred_reg = regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nMSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

with open("results/regression_results.txt", "w") as f:
    f.write("=== Resultados de Regresión ===\n")
    f.write(f"MSE: {mse:.3f}\n")
    f.write(f"R²: {r2:.3f}\n")


# ================================
# Parte 2: Clasificación con Random Forest
# ================================

print("\n=== PARTE 2: CLASIFICACIÓN ===")

y_clf = df["quality"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_clf, y_train_clf)

y_pred_clf = classifier.predict(X_test_clf)

report = classification_report(y_test_clf, y_pred_clf)
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write("=== Reporte de Clasificación ===\n")
    f.write(report)
