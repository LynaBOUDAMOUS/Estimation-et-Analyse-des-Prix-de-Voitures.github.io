# main.py
from data_preprocessing import load_data, add_features, clean_data, save_csv, plot_eda, correlation_heatmap

# Étape 1 : Charger les données
df = load_data("data/autoscrap_FIN.json")

# Étape 2 : Ajouter des features
df = add_features(df)

# Étape 3 : Nettoyer les données
df_clean = clean_data(df)

# Étape 4 : Visualisation exploratoire
plot_eda(df_clean)
correlation_heatmap(df_clean)

# Étape 5 : Export CSV pour ML ou application
save_csv(df_clean, "data/autoscrap_FIN_clean.csv")
