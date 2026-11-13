"""Chargement, nettoyage et pipeline de prétraitement des données."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    BINARY_CATEGORIES,
    BINARY_FEATURES,
    DATA_PATH,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
    TARGET,
    TARGET_MAP,
)


def _load_data(path=DATA_PATH):
    """Charge le jeu de données brut."""
    return pd.read_csv(path)


def _clean_data(df):
    """Corrige l'incohérence écran/réseaux et retire l'identifiant."""
    df = df.copy()
    df["Screen_Time_Hours"] = np.maximum(
        df["Screen_Time_Hours"], df["Social_Media_Hours"]
    )
    return df.drop(columns=["Student_ID"], errors="ignore")


def build_preprocessor():
    """Construit le ColumnTransformer : un traitement adapté à chaque type."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=BINARY_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("nom", nominal_transformer, NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(df=None):
    """Retourne X (variables brutes) et y (cible encodée 0/1/2)."""
    if df is None:
        df = _load_data()
    df = _clean_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map(TARGET_MAP)
    return X, y
