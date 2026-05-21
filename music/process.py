import numpy as np
import pandas as pd
from music.config import GENRES_DIR, GENRE_NAMES, PROCESSED_DIR


SKEWED_COLS = [
    "Spotify Followers Total",
    "Spotify Streams Total",
    "Spotify Monthly Listeners Total",
    "Spotify Playlist Reach Total",
    "YouTube Subscribers Total",
    "YouTube Views Total",
    "YouTube Likes Total",
    "TikTok Followers Total",
    "Instagram Followers Total",
    "Facebook Followers Total",
    "Deezer Fans Total",
    "SoundCloud Followers Total",
]


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


def add_rank_tier(df: pd.DataFrame) -> pd.DataFrame:
    def tier(rank):
        if rank <= 10:
            return 1
        if rank <= 50:
            return 2
        return 3
    df["rank_tier"] = df["Rank"].apply(tier)
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in SKEWED_COLS:
        if col in df.columns:
            df[f"{col} Log"] = np.log1p(df[col])
    return df


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "artists_clean.csv"
    df = pd.read_csv(path)
    assert "rank_tier" in df.columns, "rank_tier column missing"
    assert "Genre" in df.columns, "Genre column missing"
    return df


def main():
    df = load_raw()
    df = clean(df)
    df = add_rank_tier(df)
    df = add_log_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "artists_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(f"rank_tier distribution:\n{df['rank_tier'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
