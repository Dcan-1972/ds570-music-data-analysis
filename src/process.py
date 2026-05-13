import pandas as pd
from src.config import GENRES_DIR, GENRE_NAMES, PROCESSED_DIR


def load_raw() -> pd.DataFrame:
    frames = []
    for genre in GENRE_NAMES:
        path = GENRES_DIR / genre / f"{genre}-Artist-Chart-Current_Total.csv"
        df = pd.read_csv(path)
        df["Genre"] = genre
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def main():
    df = load_raw()
    df = clean(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "artists_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
