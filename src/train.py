import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.process import load_processed

TARGET = "rank_tier"
LEAKAGE_COLS = [
    "Rank",
    "Viberate Rank",
    "Spotify Rank",
    "YouTube Rank",
    "Social Rank",
    "Radio Airplay Rank",
    "Beatport Rank",
]
DROP_COLS = LEAKAGE_COLS + ["Artist Name", "Label", "rank_tier"]
CATEGORICAL_COLS = ["Genre", "Country"]


def build_xy(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y


def split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def baseline(X_train, y_train, X_test, y_test) -> float:
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average="macro")


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def cross_validate(pipe: Pipeline, X, y, cv: int = 5) -> np.ndarray:
    return cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)


def main():
    df = load_processed()
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    baseline_f1 = baseline(X_train, y_train, X_test, y_test)
    print(f"Baseline (most-frequent) F1-macro: {baseline_f1:.3f}")

    pipe = build_pipeline(X_train)
    cv_scores = cross_validate(pipe, X_train, y_train)
    print(f"RF 5-fold CV F1-macro: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


if __name__ == "__main__":
    main()
