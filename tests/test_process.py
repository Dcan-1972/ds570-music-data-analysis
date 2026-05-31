import numpy as np
import pandas as pd

from music.process import add_log_features, add_rank_tier, clean


def test_rank_tier_boundaries():
    df = pd.DataFrame({"Rank": [1, 10, 11, 50, 51, 999]})
    out = add_rank_tier(df)
    assert out["rank_tier"].tolist() == [1, 1, 2, 2, 3, 3]


def test_clean_fills_numeric_missing_with_median():
    df = pd.DataFrame({"x": [1.0, 3.0, np.nan], "Genre": ["Pop", "Rock", "Metal"]})
    out = clean(df)
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[2] == 2.0  # median of [1, 3]


def test_add_log_features_creates_log_columns():
    df = pd.DataFrame({"Spotify Followers Total": [0, np.e - 1]})
    out = add_log_features(df)
    assert "Spotify Followers Total Log" in out.columns
    assert np.isclose(out["Spotify Followers Total Log"].iloc[0], 0.0)
    assert np.isclose(out["Spotify Followers Total Log"].iloc[1], 1.0)


def test_add_log_features_skips_absent_columns():
    df = pd.DataFrame({"Rank": [1, 2, 3]})
    out = add_log_features(df)
    assert "Rank Log" not in out.columns
