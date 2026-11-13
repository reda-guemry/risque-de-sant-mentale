"""Entraînement des 3 modèles et sauvegarde du meilleur (F1 weighted)."""
import json
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_PATH,
    SCORING,
    TEST_SIZE,
    PARAM_GRIDS, 
)
from src.preprocessing import build_preprocessor, prepare_data


def _build_models(preprocessor):
    """Retourne les 3 pipelines (prétraitement + classifieur)."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _evaluate(estimator, X_test, y_test):
    """Calcule les métriques d'évaluation sur le jeu de test."""
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall (weighted)": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted"),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC (OVR)": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        ),
    }
    return metrics, y_pred


def train(save=True):
    """Entraîne les 3 modèles, garde le meilleur au F1 et le sauvegarde."""
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    models = _build_models(preprocessor)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_estimators = {}

    for name, pipeline in models.items():
        print(f"GridSearchCV — {name} ...")
        start = time.perf_counter()

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRIDS[name],
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        metrics, y_pred = _evaluate(search.best_estimator_, X_test, y_test)
        results[name] = {
            "best_params": search.best_params_,
            "cv_f1_weighted": round(float(search.best_score_), 4),
            "test_metrics": {key: round(float(value), 4) for key, value in metrics.items()},
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
            "train_seconds": round(time.perf_counter() - start, 1),
        }
        best_estimators[name] = search.best_estimator_

        print(
            f"CV F1 = {results[name]['cv_f1_weighted']:.4f} | "
            f"test F1 = {results[name]['test_metrics']['F1 (weighted)']:.4f}"
        )

    # Sélection sur le F1 de validation croisée (le test reste une évaluation honnête)
    best_name = max(results, key=lambda name: results[name]["cv_f1_weighted"])
    print(f"\nMeilleur modèle (F1 CV) : {best_name}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_estimators[best_name], BEST_MODEL_PATH)

        summary = {
            "best_model": best_name,
            "best_model_path": str(BEST_MODEL_PATH),
            "results": results,
        }
        RESULTS_PATH.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Modèle sauvegardé : {BEST_MODEL_PATH}")
        print(f"Résultats sauvegardés : {RESULTS_PATH}")

    return results, best_name, best_estimators[best_name]


if __name__ == "__main__":
    train()
