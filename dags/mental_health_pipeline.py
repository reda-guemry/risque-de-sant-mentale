"""DAG Airflow : vérification des données → entraînement → validation."""

import json
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Rend le package `src` importable quand Airflow exécute le DAG
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, MIN_F1_THRESHOLD, RESULTS_PATH


def check_data():
    """Vérifie que le fichier de données existe et contient la cible."""
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    if len(df) < 100:
        raise ValueError(f"Jeu de données trop petit : {len(df)} lignes")
    if "Mental_Health_Risk" not in df.columns:
        raise ValueError("Colonne cible 'Mental_Health_Risk' absente")
    print(f"Données OK : {df.shape[0]} lignes * {df.shape[1]} colonnes")


def train_model():
    """Lance l'entraînement et sauvegarde le meilleur modèle (F1)."""
    from src.train import train

    results, best_name, _ = train(save=True)
    print(f"Meilleur modèle : {best_name}")
    return best_name


def validate_model():
    """Vérifie que le meilleur modèle dépasse le seuil de F1."""
    summary = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    best = summary["best_model"]
    f1 = summary["results"][best]["test_metrics"]["F1 (weighted)"]
    print(f"Modèle : {best} | F1 weighted (test) = {f1:.4f}")
    if f1 < MIN_F1_THRESHOLD:
        raise ValueError(f"F1 trop faible : {f1:.4f} < {MIN_F1_THRESHOLD}")
    print("Validation OK")


with DAG(
    dag_id="student_mental_health_training",
    description="Entraînement et validation du modèle de risque en santé mentale",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "mental-health"],
) as dag:
    
    t_check = PythonOperator(
        task_id="check_data",
        python_callable=check_data
    )
    
    t_train = PythonOperator(
        task_id="train_model", 
        python_callable=train_model
    )
    
    
    t_validate = PythonOperator(
        task_id="validate_model", python_callable=validate_model
    )

    t_check >> t_train >> t_validate
