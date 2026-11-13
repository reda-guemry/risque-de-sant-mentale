"""Chargement du meilleur modèle et prédictions."""
import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, TARGET_NAMES


def load_model(path=BEST_MODEL_PATH):
    """Charge le pipeline sauvegardé (prétraitement + classifieur)."""
    return joblib.load(path)


def predict_proba(model, data):
    """Retourne les probabilités par classe pour une ligne ou un DataFrame."""
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    return model.predict_proba(data)


def predict(model, data):
    """Retourne les labels prédits (Low / Medium / High)."""
    probabilities = predict_proba(model, data)
    return [TARGET_NAMES[index] for index in probabilities.argmax(axis=1)]
