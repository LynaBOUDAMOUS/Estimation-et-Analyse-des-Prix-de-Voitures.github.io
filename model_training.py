# model_training.py

from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===========================
# PATHS
# ===========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "autoscrap_FIN_clean.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ===========================
# LOG
# ===========================
def log(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ===========================
# DATA PREPARATION
# ===========================
def prepare_data(df, target="prix", test_size=0.2):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    target = target.lower()

    if target not in df.columns:
        raise ValueError(f"❌ Target manquante : {target}")

    # Colonnes utilisées
    num_cols = ["kilometrage", "puissance_cv", "age_voiture"]
    cat_cols = ["carburant", "boite_vitesse", "marque", "modele"]

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Encodage EXACT utilisé ensuite dans l'app
    X = pd.get_dummies(
        df[num_cols + cat_cols],
        columns=cat_cols,
        drop_first=True
    )

    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns.tolist(),
        num_cols,
        cat_cols
    )

# ===========================
# METRICS
# ===========================
def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }

# ===========================
# MAIN
# ===========================
def main():

    if not DATA_PATH.exists():
        print(f"❌ CSV introuvable : {DATA_PATH}")
        sys.exit(1)

    log("📥 Chargement des données")
    df = pd.read_csv(DATA_PATH, sep=";")

    log("🧱 Préparation des données")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        num_cols,
        cat_cols
    ) = prepare_data(df)

    models = {
        "linear_regression": LinearRegression(),
        "lasso": Lasso(alpha=0.1, random_state=42),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    gbr_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    results = []
    all_params = {}
    best_model = None
    best_name = None
    best_score = float("inf")
    best_result = None

    # ===========================
    # TRAIN LOOP
    # ===========================
    for name, model in models.items():
        log(f"🚀 Entraînement : {name}")

        if name == "gradient_boosting":
            grid = GridSearchCV(
                model,
                gbr_grid,
                scoring="neg_mean_absolute_error",
                cv=3,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            params = grid.best_params_
        else:
            model.fit(X_train, y_train)
            params = model.get_params()

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        row = {"model": name, **metrics}
        results.append(row)
        all_params[name] = params

        # Sauvegarde du modèle individuel
        joblib.dump(
            {
                "model": model,
                "features": features,
                "num_cols": num_cols,
                "cat_cols": cat_cols
            },
            MODELS_DIR / f"{name}.joblib"
        )

        if metrics["mae"] < best_score:
            best_score = metrics["mae"]
            best_model = model
            best_name = name
            best_result = row

    # ===========================
    # SAVE FILES
    # ===========================
    log("💾 Sauvegarde des résultats")

    metrics_df = pd.DataFrame(results).sort_values("mae")
    metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)

    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "name": best_name
        },
        MODELS_DIR / "best_model.joblib"
    )

    with open(MODELS_DIR / "all_models_params.json", "w") as f:
        json.dump(all_params, f, indent=2)

    with open(MODELS_DIR / "best_model_result.json", "w") as f:
        json.dump(best_result, f, indent=2)

    log("🏁 RÉSUMÉ FINAL")
    print(metrics_df.to_string(index=False))
    print(f"\n🏆 Meilleur modèle : {best_name}")
    print(best_result)

# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()
