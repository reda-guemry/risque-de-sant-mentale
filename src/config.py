"""Configuration centralisée du projet (chemins, cible, variables)."""
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "student_mental_health.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RESULTS_PATH = MODELS_DIR / "training_results.json"

# Cible
TARGET = "Mental_Health_Risk"
TARGET_NAMES = ["Low", "Medium", "High"]
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Entraînement
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1_weighted"
MIN_F1_THRESHOLD = 0.70  # seuil utilisé par la validation du DAG Airflow

# Groupes de variables (utilisés par le ColumnTransformer)
NUMERIC_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]
ORDINAL_FEATURES = ["Physical_Activity", "Diet_Quality"]
ORDINAL_CATEGORIES = [["Low", "Moderate", "High"], ["Poor", "Average", "Good"]]
BINARY_FEATURES = ["Part_Time_Job", "Previous_Mental_Health_History"]
BINARY_CATEGORIES = [["No", "Yes"], ["No", "Yes"]]
NOMINAL_FEATURES = ["Gender"]


# Paramètres de recherche pour chaque modèle
PARAM_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    },
    "XGBoost": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
    },
}

