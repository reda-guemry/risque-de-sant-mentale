"""Chargement du meilleur modèle et prédictions."""
import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, TARGET_NAMES


def load_model(path=BEST_MODEL_PATH):
    """Charge le pipeline sauvegardé (prétraitement + classifieur)."""
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Impossible de charger le modèle '{path}' : le module "
            f"'{exc.name}' est absent de l'environnement. "
            "Reconstruisez l'image avec `docker compose up -d --build` "
            "ou réentraînez le modèle avec `docker compose run --rm trainer`."
        ) from exc


def predict_proba(model, data):
    """Retourne les probabilités par classe pour une ligne ou un DataFrame."""
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    return model.predict_proba(data)


def predict(model, data):
    """Retourne les labels prédits (Low / Medium / High)."""
    probabilities = predict_proba(model, data)
    return [TARGET_NAMES[index] for index in probabilities.argmax(axis=1)]
