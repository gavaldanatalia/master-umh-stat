#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:55:20 2025

@author: jjmilla
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Cargar dataset
def load_data(file_path, target_column, problem_type="classification"):
    """
    Carga un dataset desde un archivo CSV y lo divide en características (X) y etiquetas (y).
    
    Args:
        file_path (str): Ruta al archivo CSV.
        target_column (str): Nombre de la columna objetivo.
        problem_type (str): Tipo de problema ("classification" o "regression").
    
    Returns:
        X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba.
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if problem_type == "classification":
        y = y.astype("category").cat.codes  # Convertir a valores numéricos si es clasificación
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar árbol de decisión para clasificación
def train_decision_tree_classifier(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

# Crear y entrenar árbol de decisión para regresión
def train_decision_tree_regressor(X_train, y_train):
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    return reg

# Evaluar modelos
def evaluate_model(model, X_test, y_test, problem_type="classification"):
    y_pred = model.predict(X_test)
    if problem_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo de clasificación: {acc:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        print(f"Error cuadrático medio (MSE) del modelo de regresión: {mse:.2f}")

# Main
if __name__ == "__main__":
    file_path = "dataset.csv"  # Reemplaza con la ruta de tu dataset
    target_column = "target"   # Reemplaza con el nombre de tu columna objetivo

    # Clasificación
    X_train, X_test, y_train, y_test = load_data(file_path, target_column, problem_type="classification")
    classifier = train_decision_tree_classifier(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, problem_type="classification")

    # Regresión
    X_train, X_test, y_train, y_test = load_data(file_path, target_column, problem_type="regression")
    regressor = train_decision_tree_regressor(X_train, y_train)
    evaluate_model(regressor, X_test, y_test, problem_type="regression")
