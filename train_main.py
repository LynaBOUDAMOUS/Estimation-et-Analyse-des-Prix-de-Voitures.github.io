# train_main.py
import pandas as pd
from data_preprocessing import load_data, add_features, clean_data
from model_training import prepare_data, train_models, evaluate_models, save_best_model
from data_preprocessing import load_data, add_features, clean_data, process_nom_annonce
from model_training import prepare_data, train_models, evaluate_models, save_best_model

# Étape 1 : Charger les données
df = load_data("data/autoscrap_FIN.json")

# Étape 1b : Traiter nom_annonce pour créer marque et modele
df = process_nom_annonce(df)

# Étape 2 : Ajouter les features
df = add_features(df)

# Étape 3 : Nettoyer les données
df_clean = clean_data(df)

# Vérification des colonnes
print(df_clean.columns)  # pour s'assurer que 'marque' et 'modele' sont présentes

# Étape 4 : Préparer les données pour ML
X_train, X_test, y_train, y_test = prepare_data(df_clean, target='prix')

# Étape 5 : Entraîner les modèles
models = train_models(X_train, y_train)

# Étape 6 : Évaluer les modèles
results = evaluate_models(models, X_test, y_test)
print("Évaluation des modèles :")
print(results)

# Étape 7 : Sauvegarder le meilleur modèle
best_model_name = save_best_model(models, X_test, y_test, path="models/best_model.pkl")
print(f"Le meilleur modèle est : {best_model_name}")