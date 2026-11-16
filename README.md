# 🧠 Risque de santé mentale — Étudiants

Prédiction du niveau de risque (`Low` / `Medium` / `High`) à partir de variables
comportementales, académiques et psychosociales.

## Structure

```
data/                             jeu de données
notebooks/                        EDA et expérimentations
src/                              code source réutilisable
  config.py                       chemins, cible, groupes de variables
  preprocessing.py                nettoyage + ColumnTransformer
  train.py                        3 modèles + GridSearchCV + sauvegarde du meilleur
  predict.py                      chargement du modèle et prédictions
app/streamlit_app.py              interface Streamlit
dags/mental_health_pipeline.py    DAG Airflow (check → train → validate)
models/                           modèle sauvegardé + résultats
Dockerfile                        images trainer / streamlit / airflow
docker-compose.yml                orchestration des services
```

## 🐳 Démarrage avec Docker (recommandé)

Services séparés : `streamlit`, `trainer`, `airflow-apiserver`,
`airflow-scheduler`, `airflow-dag-processor` et `postgres`.

```bash
# 1. Construire et démarrer l'interface + la stack Airflow
docker compose up -d --build

# 2. Entraîner le modèle (écrit dans le volume partagé models_data)
docker compose run --rm trainer
```

- Streamlit : http://localhost:8501
- Airflow : http://localhost:8080 — identifiants `admin` / `admin`

Le volume `models_data` est partagé entre les services : le DAG Airflow peut
réentraîner le modèle et Streamlit le recharge au prochain rafraîchissement.

Pour déclencher le DAG : dans l'UI Airflow, activer puis déclencher
`student_mental_health_training`, ou en ligne de commande :

```bash
docker compose exec airflow-scheduler airflow dags unpause student_mental_health_training
```

Arrêt : `docker compose down` (ajouter `-v` pour supprimer aussi les volumes).

> Le premier build est long (Airflow + dépendances ML).

## 💻 Utilisation locale (sans Docker)

### Installation

```bash
uv sync
```

### Entraînement

```bash
python -m src.train
```

Le script entraîne **Logistic Regression**, **Random Forest** et **XGBoost**
avec `GridSearchCV` (5 folds stratifiés, `f1_weighted`), puis sauvegarde
**uniquement le meilleur modèle** :

- `models/best_model.joblib` — pipeline complet (prétraitement + classifieur)
- `models/training_results.json` — hyperparamètres, scores CV et métriques de test

### Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Orchestration Airflow

Airflow est fourni en groupe de dépendances séparé :

```bash
uv sync --group airflow
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export PYTHONPATH="$(pwd)"
uv run airflow db migrate
uv run airflow standalone
```

Le DAG `student_mental_health_training` s'exécute chaque semaine et enchaîne :

1. `check_data` — vérifie le fichier et la colonne cible ;
2. `train_model` — entraîne les 3 modèles et sauvegarde le meilleur ;
3. `validate_model` — refuse le pipeline si le F1 weighted est < `0.70`.

> En cas de conflit de dépendances, utilisez Docker ou un environnement
> virtuel dédié pour Airflow.
