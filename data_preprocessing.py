# data_preprocessing.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")  # Style global pour plots

# ---------------------------
# Chargement et features
# ---------------------------

def load_data(json_path: str) -> pd.DataFrame:
    """Charger les données depuis un fichier JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajouter des colonnes dérivées comme l'âge de la voiture."""
    df = df.copy()
    df['age_voiture'] = 2026 - df['annee']
    return df

# ---------------------------
# Traitement nom_annonce
# ---------------------------

def process_nom_annonce(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renommer la colonne 'nom_annonce' et créer deux nouvelles colonnes :
    - 'marque' : la première partie du nom
    - 'modele' : le reste du nom
    """
    df = df.copy()
    
    if 'nom_annonce' in df.columns:
        # Séparer la première partie (marque) et le reste (modèle)
        df[['marque', 'modele']] = df['nom_annonce'].str.split(pat=' ', n=1, expand=True)

        # Supprimer la colonne originale
        df = df.drop(columns=['nom_annonce'])
    
    return df

# ---------------------------
# Détection et nettoyage outliers
# ---------------------------

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Identifier les outliers selon la méthode IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyer les données en retirant les outliers et en convertissant les types."""
    df = df.copy()
    
    outliers_prix = detect_outliers_iqr(df['prix'])
    outliers_kilometrage = detect_outliers_iqr(df['kilometrage'])
    outliers_puissance = detect_outliers_iqr(df['puissance_cv'])
    
    df_clean = df[
        (~df['prix'].isin(outliers_prix)) &
        (~df['kilometrage'].isin(outliers_kilometrage)) &
        (~df['puissance_cv'].isin(outliers_puissance))
    ].copy()
    
    # Conversion code postal en int
    df_clean['code_postal'] = df_clean['code_postal'].astype(int)
    
    return df_clean

# ---------------------------
# Export CSV
# ---------------------------

def save_csv(df: pd.DataFrame, path: str):
    """Exporter le DataFrame nettoyé en CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";")
    print(f"Le fichier CSV a été créé avec succès : {path}")

# ---------------------------
# Visualisation exploratoire
# ---------------------------

def plot_eda(df: pd.DataFrame):
    """Fonctions pour visualisation exploratoire."""
    # Boxplots
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    sns.boxplot(y=df['prix'])
    plt.title('Boxplot Prix (€)')
    
    plt.subplot(1,3,2)
    sns.boxplot(y=df['kilometrage'])
    plt.title('Boxplot Kilométrage (km)')
    
    plt.subplot(1,3,3)
    sns.boxplot(y=df['puissance_cv'])
    plt.title('Boxplot Puissance (CV)')
    
    plt.tight_layout()
    plt.show()
    
    # Histogrammes
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    sns.histplot(df['prix'], bins=30, kde=True)
    plt.title('Histogramme Prix (€)')
    
    plt.subplot(1,3,2)
    sns.histplot(df['kilometrage'], bins=30, kde=True)
    plt.title('Histogramme Kilométrage (km)')
    
    plt.subplot(1,3,3)
    sns.histplot(df['puissance_cv'], bins=30, kde=True)
    plt.title('Histogramme Puissance (CV)')
    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame):
    """Afficher la corrélation entre variables numériques."""
    plt.figure(figsize=(10,8))
    corr = df[['prix','annee','kilometrage','puissance_cv','age_voiture']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, mask=mask,
                square=True, linewidths=0.5, cbar_kws={"shrink":0.8})
    plt.title("Corrélation entre variables numériques", fontsize=16)
    plt.show()

# ---------------------------
# Exemple d'utilisation
# ---------------------------
if __name__ == "__main__":
    df = load_data("data/autoscrap_FIN.json")
    df = process_nom_annonce(df)  # traitement nom_annonce
    df = add_features(df)
    df_clean = clean_data(df)
    save_csv(df_clean, "data/autoscrap_FIN_clean.csv")
    print(df_clean.head())
