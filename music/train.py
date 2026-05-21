import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from music.config import OUTPUTS_DIR
from music.process import load_processed

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


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    rf: RandomForestClassifier = pipe.named_steps["rf"]
    names = pre.get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_chart(imp: pd.DataFrame, top_n: int = 15) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    top = imp.head(top_n)
    fig = px.bar(
        top.iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances — Random Forest",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    out = OUTPUTS_DIR / "feature_importance.html"
    fig.write_html(out)
    print(f"Saved feature importance chart to {out}")


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

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"Test F1-macro: {f1_score(y_test, preds, average='macro'):.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=3))

    imp = feature_importance(pipe)
    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))
    save_feature_importance_chart(imp)

    model_path = OUTPUTS_DIR / "model.pkl"
    joblib.dump(pipe, model_path)
    print(f"\nSaved trained model to {model_path}")


if __name__ == "__main__":
    main()
