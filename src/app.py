import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


## Sin feature selection

BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

results = []
models=[]

for index, dataset in enumerate(TRAIN_DATASETS):
    model = RandomForestClassifier(random_state = 42)
    model.fit(dataset, y_train.values.ravel())
    models.append(model)
    
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train": accuracy_score(y_train, y_pred_train),
            "test": accuracy_score(y_test, y_pred_test)
        }
    )


## Con feature selection
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
    'n_estimators': [50, 100, 200, 300],            
    'max_depth': [None, 8, 12, 16],                 
    'min_samples_split': [2, 5, 10],                
    'min_samples_leaf': [1, 2, 4],                  
    'bootstrap': [True],                            
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
    bootstrap=True,
    criterion='entropy',
    max_depth=None,
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=100,
    random_state=42
)

final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print(f"Train accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_test)}")

## Guardando el modelo

dump(model, open("../models/random_forest_classifier_42.sav", "wb"))

