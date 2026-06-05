# Artist Rank Predictor — DS570 Final Project

Predicting an artist's per-genre rank tier (Top 10 / Top 11-50 / Top 51+) from
cross-platform music metrics — Spotify, YouTube, TikTok, Instagram, Deezer, and
more.

The goal is to identify which platforms actually drive top-tier status, and to
ship the analysis as an interactive Streamlit dashboard.

## Problem

Music industry chart positions are widely tracked, but the underlying signal is
opaque: which platform metrics most strongly predict whether an artist is a
genre-level superstar? This project frames that as a 3-class classification
problem on per-genre Viberate ranks.

**Why per-genre rather than global rank?** Top performers across genres are not
directly comparable — the global Viberate rank is dominated by Pop and Hip Hop.
Per-genre tiers ask the more useful question: *what makes someone top-tier
within their genre?*

## Data

- **Source:** [Viberate](https://viberate.com/) — Artist Chart "Current Total"
  files for five genres (Pop, Rock, Hip Hop, Electronic, Metal). Data was
  exported via a Viberate account.
- **Snapshot in repo:** `Assets/` contains 2,500 artists × ~27 columns
  (rank, country, label, Spotify/YouTube/TikTok/Instagram/Facebook/Deezer/
  SoundCloud/Beatport metrics). The snapshot is committed to this public
  repository so the project is fully reproducible without any login.
- **What the pipeline actually uses:** only the five `Genres/<genre>/
  <genre>-Artist-Chart-Current_Total.csv` files are read by `process.py`. The
  `Assets/` folder also bundles Countries, Festivals, and Playlists exports;
  these are kept for reference and possible future extensions but are not part
  of the current model.
- **Why bundled instead of fetched at runtime:** Viberate's chart export sits
  behind an account, so a live API/cloud fetch would force a sign-up — exactly
  what the guidelines forbid. Instead a one-time static snapshot (~34 MB) is
  committed to the repo and `COPY`-ed into the image. The grader only runs
  `docker build` + `docker run`: no download, no login, and the build stays
  well under the time limit.
- **Terms of use / licensing:** the data is Viberate's proprietary chart data,
  used here strictly for non-commercial educational coursework. It is not
  redistributed for any commercial purpose and is included only so this academic
  project is self-contained and reproducible by the instructor and classmates.
- **Target:** `rank_tier` derived from per-genre `Rank` — 1 = Top 10,
  2 = Top 11-50, 3 = Top 51+. Class distribution: 50 / 200 / 2250.

## Methods

- **Processing** (`music/process.py`): concatenate five genre files, median
  impute numeric columns, derive `rank_tier`, add `log1p` features for skewed
  follower/stream counts. Both the raw and log-scaled versions are kept as
  features — they are collinear, which a Random Forest tolerates fine, but it
  means raw and log of the same metric (e.g. Deezer Fans) can both appear near
  the top of the importance ranking.
- **Pipeline** (`music/train.py`): `ColumnTransformer` with `StandardScaler`
  for numeric features and `OneHotEncoder` for `Genre` and `Country`, feeding a
  `RandomForestClassifier(n_estimators=200, class_weight="balanced")`.
- **Why Random Forest:** the features are highly skewed, on wildly different
  scales, and interact non-linearly (a huge TikTok following means something
  different for a Metal artist than a Pop artist). A tree ensemble handles all
  of this without heavy distributional assumptions, gives feature importances
  for free, and supports `class_weight="balanced"` for the imbalance. Logistic
  regression was a weaker fit (it assumes a linear log-odds relationship and is
  sensitive to the collinear raw/log features); a single decision tree would
  overfit; gradient boosting was avoided to keep training fast and the model
  easy to explain in the oral demo.
- **No data leakage:** every column that encodes the answer is dropped before
  training (`LEAKAGE_COLS` in `train.py` — `Rank`, `Viberate Rank`, and the
  per-platform rank columns), since the target is derived from `Rank`. All
  preprocessing lives inside the `Pipeline`, so scaling/encoding are fit on the
  training fold only and never see the test data.
- **Evaluation:** stratified 80/20 train/test split, 5-fold cross-validation
  on the training set, F1-macro as the primary metric (the right choice given
  the heavy class imbalance), plus a `DummyClassifier(strategy="most_frequent")`
  baseline. Feature importance is read from the fitted Random Forest.

## Results

| Metric | Value |
|---|---|
| Baseline F1-macro (most-frequent class) | 0.316 |
| 5-fold CV F1-macro | 0.692 ± 0.033 |
| Test F1-macro | 0.705 |
| Test accuracy | 0.944 |

The Random Forest more than doubles the baseline F1-macro, which is the
meaningful comparison on a 90/8/2 imbalanced problem (accuracy alone would
look misleadingly high for the dummy model).

**Top features by importance:** Deezer Fans (raw and log-scaled), Spotify
Followers and Streams, YouTube Subscribers, Spotify Monthly Listeners. Deezer
is a surprisingly strong signal — likely because top-tier artists are far more
likely to have any Deezer presence at all.

## Dashboard

A three-tab Streamlit dashboard (`app/dashboard.py`):

- **EDA** — artist counts per genre (with a genre filter) and a Spotify-streams
  vs TikTok-followers scatter (log-log, colored by genre).
- **ML Results** — metric tiles (baseline, CV, test F1-macro, test accuracy),
  top-15 feature importance chart, and a prediction form: enter genre,
  country, and key platform metrics to get a predicted rank tier and class
  probabilities.
- **About** — short project summary.

## Repository layout

```
ds570-music-data-analysis/
├── Assets/                    raw Viberate exports (pipeline uses 5 genre files)
├── music/
│   ├── config.py              path constants
│   ├── process.py             load, clean, feature-engineer
│   └── train.py               train and evaluate the RF model
├── app/
│   └── dashboard.py           Streamlit app
├── tests/
│   └── test_process.py        unit tests for the processing functions
├── notebooks/
│   └── 2026-05-10-dcb-Genre-EDA.ipynb
├── data/processed/            generated by process-data (gitignored)
├── outputs/                   generated by train-model (gitignored)
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Build and run

### Local

```bash
pip install -e .
process-data
train-model
streamlit run app/dashboard.py
```

Then open http://localhost:8501.

### Docker

```bash
docker build -t music-app .
docker run -p 8501:8501 music-app
```

Then open http://localhost:8501. The image bundles the data and bakes the
processed dataset and trained model in at build time, so the dashboard starts
immediately.

## Limitations

- **Imbalanced classes (50 / 200 / 2250):** the Top-51+ tier is essentially
  solved (per-class F1 ≈ 0.98), but the two minority tiers are much harder —
  on the held-out test set the Top-10 tier is recalled ~40% and the middle tier
  (Top 11-50) ~50%. So the model *does* recover the middle tier about half the
  time; the limit is the small number of top-tier artists, not a complete failure
  to learn them. The prediction form's class-probability bars expose this
  uncertainty honestly rather than hiding it behind a single label.
- **What did and didn't help (tested):** adding cross-platform **momentum**
  features from the 30d/3m/12m chart exports (month-over-month growth columns)
  did *not* meaningfully move F1-macro (~0.005), which points to class scarcity
  rather than feature poverty as the bottleneck. The one change that does lift the
  score is reframing the **target**: collapsing to a binary Top-50 / Top-51+ split
  reaches F1-macro ≈ 0.83 — but that answers an easier question (it drops the
  Top-10 vs Top-11-50 distinction), so the more informative 3-class target is kept
  as the headline model. Oversampling (e.g. SMOTE) is another option left out of
  scope here.
- **Cross-sectional snapshot:** the data is a single point in time. The model
  predicts rank tier from current metrics; it doesn't capture momentum or
  career trajectory.
- **Per-genre, not cross-genre:** the model is trained on each genre's own
  Top-500 chart, so "tier 1" in Metal is not the same global popularity as
  "tier 1" in Pop. This is a deliberate framing choice (see Problem above).

## Course

Built for DS570 — Introduction to Data Science.
