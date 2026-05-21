import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.process import load_processed

TARGET = "rank_tier"
DROP_COLS = ["Rank", "Viberate Rank", "Artist", "rank_tier"]


def build_xy(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.select_dtypes(include=["number", "object"])
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


def main():
    df = load_processed()
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"Train tier distribution:\n{y_train.value_counts().sort_index()}")
    f1 = baseline(X_train, y_train, X_test, y_test)
    print(f"Baseline (most-frequent) F1-macro: {f1:.3f}")


if __name__ == "__main__":
    main()
