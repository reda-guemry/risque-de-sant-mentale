"""Interface Streamlit simple de prédiction du risque en santé mentale."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BEST_MODEL_PATH, RESULTS_PATH, TARGET_NAMES
from src.predict import load_model, predict_proba

st.set_page_config(page_title="Risque santé mentale", layout="centered")
st.title("Prédiction du risque en santé mentale")
st.write("Renseignez le profil de l'étudiant pour estimer son niveau de risque.")

if not BEST_MODEL_PATH.exists():
    st.error("Aucun modèle trouvé. Lancez d'abord `python -m src.train`.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

if RESULTS_PATH.exists():
    summary = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    best = summary["best_model"]
    f1 = summary["results"][best]["test_metrics"]["F1 (weighted)"]
    st.caption(f"Modèle utilisé : **{best}** — F1 weighted (test) : **{f1:.3f}**")

with st.form("profil_etudiant"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=17, max_value=40, value=21)
        gender = st.selectbox("Genre", ["Male", "Female", "Non-binary"])
        year = st.selectbox("Année d'études", [1, 2, 3, 4, 5], index=2)
        sleep = st.slider("Sommeil (h / nuit)", 3.0, 10.0, 7.0, 0.1)
        study = st.number_input(
            "Étude (h / semaine)", min_value=0.0, max_value=60.0, value=20.0, step=0.5
        )
        gpa = st.slider("GPA (sur 4)", 1.5, 4.0, 2.6, 0.01)
        social = st.slider("Réseaux sociaux (h / jour)", 0.0, 12.0, 6.0, 0.1)
        screen = st.slider("Temps d'écran total (h / jour)", 1.0, 14.0, 8.5, 0.1)

    with col2:
        activity = st.selectbox("Activité physique", ["Low", "Moderate", "High"], index=1)
        financial = st.slider("Stress financier (1-10)", 1.0, 10.0, 6.0, 0.1)
        family = st.slider("Soutien familial (1-10)", 1.0, 10.0, 6.5, 0.1)
        attendance = st.slider("Assiduité (%)", 30.0, 100.0, 85.0, 0.5)
        job = st.selectbox("Emploi à temps partiel", ["No", "Yes"])
        diet = st.selectbox("Qualité de l'alimentation", ["Poor", "Average", "Good"], index=1)
        history = st.selectbox("Antécédents de santé mentale", ["No", "Yes"])

    submitted = st.form_submit_button("Prédire le risque")

if submitted:
    student = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year,
                "Sleep_Hours": sleep,
                "Study_Hours_Per_Week": study,
                "GPA": gpa,
                "Social_Media_Hours": social,
                "Screen_Time_Hours": screen,
                "Physical_Activity": activity,
                "Financial_Stress": financial,
                "Family_Support": family,
                "Attendance_Rate": attendance,
                "Part_Time_Job": job,
                "Diet_Quality": diet,
                "Previous_Mental_Health_History": history,
            }
        ]
    )

    probabilities = predict_proba(model, student)[0]
    risk = TARGET_NAMES[int(probabilities.argmax())]

    if risk == "Low":
        st.success(f"### Risque prédit : {risk}")
    elif risk == "Medium":
        st.warning(f"### Risque prédit : {risk}")
    else:
        st.error(f"### Risque prédit : {risk}")

    st.bar_chart(pd.DataFrame({"Probabilité": probabilities}, index=TARGET_NAMES))
    st.dataframe(
        pd.DataFrame({"Classe": TARGET_NAMES, "Probabilité": probabilities.round(3)})
    )
