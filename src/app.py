import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

## Cargar el dataset
train_data = pd.read_csv("../data/processed/clean_train_con_outliers.csv")
test_data = pd.read_csv("../data/processed/clean_test_con_outliers.csv")

X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]

## Modelo y predicciones
model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

## Hiperparametrización

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 100, 150],
    'max_depth': [4, 6, 8, None],
    'bootstrap':[True, False],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'criterion': ['gini', 'entropy']
}



grid = GridSearchCV(model, param_grid, scoring = "accuracy", cv = 5)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")

final_model = RandomForestClassifier(
    bootstrap=False,
    criterion='entropy',
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=2,
    n_estimators=40,
    random_state=42
)

final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print(f"Train accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_test)}")

## Guardando el modelo

dump(model, open("../models/random_forest_classifier_42.sav", "wb"))

