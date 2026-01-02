import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------
# Chargement des ressources
# ---------------------------

@st.cache_data
def load_data(path="data/autoscrap_FIN_clean.csv"):
    return pd.read_csv(path, sep=";")

@st.cache_resource
def load_model(path="models/best_model.joblib"):
    bundle = joblib.load(path)
    return (
        bundle["model"],
        bundle["features"],
        bundle["num_cols"],
        bundle["cat_cols"]
    )

def prepare_input(df_train, user_input, features, cat_cols):
    input_df = pd.DataFrame([user_input])

    combined = pd.concat([df_train, input_df], axis=0)

    encoded = pd.get_dummies(
        combined,
        columns=cat_cols,
        drop_first=True
    )

    # Ajouter les colonnes manquantes
    for col in features:
        if col not in encoded.columns:
            encoded[col] = 0

    # Conserver uniquement les colonnes du modèle
    encoded = encoded[features]

    return encoded.tail(1)

# ---------------------------
# UI
# ---------------------------

st.title("🚗 Prédiction du prix d'une voiture d'occasion")
st.write("Entrez les caractéristiques pour estimer le prix.")

df = load_data()
model, features, num_cols, cat_cols = load_model()

st.sidebar.header("Caractéristiques")

kilometrage = st.sidebar.number_input("Kilométrage (km)", 0, 300_000, 50_000, 1_000)
puissance_cv = st.sidebar.number_input("Puissance (CV)", 20, 500, 100)
age_voiture = st.sidebar.number_input("Âge (années)", 0, 50, 5)

carburant = st.sidebar.selectbox("Carburant", sorted(df["carburant"].unique()))
boite = st.sidebar.selectbox("Boîte de vitesse", sorted(df["boite_vitesse"].unique()))
marque = st.sidebar.selectbox("Marque", sorted(df["marque"].unique()))
modele = st.sidebar.selectbox("Modèle", sorted(df["modele"].unique()))

user_input = {
    "kilometrage": kilometrage,
    "puissance_cv": puissance_cv,
    "age_voiture": age_voiture,
    "carburant": carburant,
    "boite_vitesse": boite,
    "marque": marque,
    "modele": modele
}

if st.button("💰 Prédire le prix"):
    X = prepare_input(
        df_train=df[num_cols + cat_cols],
        user_input=user_input,
        features=features,
        cat_cols=cat_cols
    )
    prix = model.predict(X)[0]
    st.success(f"Prix estimé : **{prix:,.0f} €**")

# ---------------------------
# Visualisations
# ---------------------------

st.subheader("📊 Aperçu des données")
st.dataframe(df.sample(10))

fig1 = px.box(df, x="carburant", y="prix", title="Prix par carburant")
st.plotly_chart(fig1, width="stretch")


fig2 = px.scatter(
    df,
    x="age_voiture",
    y="prix",
    color="puissance_cv",
    title="Prix vs âge"
)
st.plotly_chart(fig2, width="stretch")

