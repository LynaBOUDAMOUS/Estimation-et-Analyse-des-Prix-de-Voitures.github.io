# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def prepare_data(df: pd.DataFrame, target: str = "prix") -> tuple:
    """
    Séparer les features de la target et encoder les variables catégorielles.
    Retourne X_train, X_test, y_train, y_test
    """
    df = df.copy()
    
    # Variables numériques
    numeric_features = ['kilometrage', 'puissance_cv', 'age_voiture']
    
    # Variables catégorielles à encoder
    categorical_features = ['carburant', 'boite_vitesse', 'marque', 'modele']
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df[numeric_features + categorical_features], drop_first=True)
    
    X = df_encoded
    y = df[target]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Entraîner plusieurs modèles et retourner un dictionnaire avec tous les modèles entraînés.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models: dict, X_test, y_test):
    """
    Évaluer les modèles sur le test set et retourner un DataFrame avec les scores RMSE et R2.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'R2': r2})
    
    return pd.DataFrame(results).sort_values(by='RMSE')

def save_best_model(models: dict, X_test, y_test, path: str = "best_model.pkl"):
    """
    Identifier le meilleur modèle selon RMSE et le sauvegarder avec joblib.
    """
    best_model_name = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
    
    joblib.dump(models[best_model_name], path)
    print(f"Meilleur modèle sauvegardé : {best_model_name} avec RMSE = {best_rmse:.2f}")
    return best_model_name

