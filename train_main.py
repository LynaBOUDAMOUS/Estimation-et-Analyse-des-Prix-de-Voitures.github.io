# train_main.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from data_preprocessing import load_data, add_features, clean_data, process_nom_annonce
from model_training import select_features, evaluate

# ---------------------------
# Chemins
# ---------------------------
DATA_PATH = Path("data/autoscrap_FIN_clean.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Étape 1 : Charger et traiter les données
# ---------------------------
df = load_data("data/autoscrap_FIN.json")
df = process_nom_annonce(df)  # créer 'marque' et 'modele'
df = add_features(df)
df_clean = clean_data(df)

print("Colonnes après nettoyage :", df_clean.columns)

# ---------------------------
# Étape 2 : Sélection des features / target
# ---------------------------
X, y, feat_cols = select_features(df_clean, target="prix")
print(f"Shape X: {X.shape}, y: {y.shape}, Nb features: {len(feat_cols)}")

# ---------------------------
# Étape 3 : Baseline (mean)
# ---------------------------
baseline_pred = np.full(len(y), fill_value=y.mean())
r = evaluate(y, baseline_pred, "baseline_mean")

# ---------------------------
# Étape 4 : Modèles
# ---------------------------
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = []
best_model = None
best_name = None
best_mae = float("inf")

for name, model in models.items():
    print(f"\nEntraînement: {name}")
    model.fit(X, y)
    y_pred = model.predict(X)
    r = evaluate(y, y_pred, name)
    results.append(r)
    if r["mae"] < best_mae:
        best_mae, best_model, best_name = r["mae"], model, name

# ---------------------------
# Étape 5 : Sauvegarde
# ---------------------------
metrics_df = pd.DataFrame(results).sort_values("mae")
metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)

joblib.dump({"model": best_model, "features": feat_cols, "name": best_name},
            MODELS_DIR / "best_model.joblib")

print("\n🏁 Résumé final")
print(metrics_df)
print(f"✅ Meilleur modèle : {best_name}, sauvegardé dans {MODELS_DIR / 'best_model.joblib'}")
