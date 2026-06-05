import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from music.config import OUTPUTS_DIR
from music.process import SKEWED_COLS, load_processed
from music.train import DROP_COLS

TIER_LABELS = {1: "Top 10", 2: "Top 11-50", 3: "Top 51+"}

st.set_page_config(
    page_title="Artist Rank Predictor",
    page_icon=":musical_note:",
    layout="wide",
)

st.title("Artist Rank Predictor")
st.caption(
    "Predicting artist rank tier from cross-platform music metrics — DS570 Final Project"
)
st.markdown(
    "<hr style='border:none;border-top:3px solid #1F6FB2;margin:0.2rem 0 0.6rem;'>",
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    return load_processed()


df = load_data()

eda_tab, ml_tab, about_tab = st.tabs(["EDA", "ML Results", "About"])

with eda_tab:
    st.header("Exploratory Data Analysis")

    genre_choice = st.selectbox(
        "Filter by genre",
        options=["All"] + sorted(df["Genre"].unique().tolist()),
    )

    if genre_choice == "All":
        view = df
    else:
        view = df[df["Genre"] == genre_choice]

    st.metric("Artists shown", len(view))

    counts = (
        view.groupby("Genre", as_index=False)
        .size()
        .rename(columns={"size": "Artist Count"})
        .sort_values("Artist Count", ascending=False)
    )

    fig = px.bar(
        counts,
        x="Genre",
        y="Artist Count",
        color="Genre",
        title="Number of Artists per Genre",
        labels={"Artist Count": "Number of Artists"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spotify vs TikTok reach")
    scatter_df = view[
        ["Artist Name", "Genre", "Spotify Streams Total", "TikTok Followers Total"]
    ].dropna()
    scatter_fig = px.scatter(
        scatter_df,
        x="Spotify Streams Total",
        y="TikTok Followers Total",
        color="Genre",
        hover_name="Artist Name",
        log_x=True,
        log_y=True,
        title="Spotify Streams vs TikTok Followers (log scale)",
        opacity=0.7,
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.caption(
        "Missing platform values were median-imputed during cleaning. That is why "
        "some points line up in a horizontal band — those artists had no TikTok "
        "data and were filled with the median follower count."
    )

with ml_tab:
    st.header("Model Results")

    metrics_path = OUTPUTS_DIR / "metrics.json"
    model_path = OUTPUTS_DIR / "model.pkl"

    if not metrics_path.exists() or not model_path.exists():
        st.warning("Model artifacts not found. Run `train-model` first.")
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

        f1_lift = metrics["test_f1_macro"] - metrics["baseline_f1_macro"]
        c1, c2, c3, c4 = st.columns(4)
        with c1.container(border=True):
            st.metric("Baseline F1-macro", f"{metrics['baseline_f1_macro']:.3f}",
                      help="Most-frequent-class DummyClassifier")
        with c2.container(border=True):
            st.metric("CV F1-macro", f"{metrics['cv_f1_macro_mean']:.3f}",
                      help=f"+/- {metrics['cv_f1_macro_std']:.3f} across 5 folds")
        with c3.container(border=True):
            st.metric("Test F1-macro", f"{metrics['test_f1_macro']:.3f}",
                      delta=f"+{f1_lift:.3f} vs baseline")
        with c4.container(border=True):
            st.metric("Test accuracy", f"{metrics['test_accuracy']:.3f}",
                      delta="headline")

        st.caption(
            f"Trained on {metrics['n_train']} artists, tested on {metrics['n_test']}. "
            "Random Forest with class_weight='balanced' on imbalanced rank tiers."
        )

        model = joblib.load(model_path)
        pre = model.named_steps["pre"]
        rf = model.named_steps["rf"]
        importance_df = (
            pd.DataFrame({
                "feature": pre.get_feature_names_out(),
                "importance": rf.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .head(15)
        )
        importance_df["feature"] = importance_df["feature"].str.replace(
            r"^(num__|cat__)", "", regex=True
        )

        imp_fig = px.bar(
            importance_df.iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        imp_fig.update_layout(height=500)
        st.plotly_chart(imp_fig, use_container_width=True)

        st.divider()
        st.subheader("Predict an artist's rank tier")
        st.caption(
            "Enter cross-platform totals. Defaults are a typical Top-10 artist — "
            "note the scale: top artists have **billions** of streams and tens of "
            "millions of followers."
        )

        genres = sorted(df["Genre"].dropna().unique().tolist())
        countries = sorted(df["Country"].dropna().unique().tolist())
        default_country = df["Country"].mode().iloc[0]

        # The model leans on all of these cross-platform totals. The form exposes
        # every one so the row it builds is internally consistent: setting only a
        # few while the rest stayed at the (low-tier) median produced misleadingly
        # pessimistic predictions even for superstar-level inputs.
        superstar = df[df["rank_tier"] == 1]

        def default_for(col):
            return int(superstar[col].quantile(0.75))

        # Group the inputs by platform (first word of the column name) so the form
        # reads as Spotify / YouTube / TikTok / ... blocks instead of one long list.
        platform_groups = {}
        for feat in SKEWED_COLS:
            platform_groups.setdefault(feat.split()[0], []).append(feat)

        with st.form("predict_form"):
            in_genre = st.selectbox("Genre", genres)
            in_country = st.selectbox(
                "Country", countries, index=countries.index(default_country)
            )
            inputs = {}
            for platform, feats in platform_groups.items():
                st.markdown(f"**{platform}**")
                form_cols = st.columns(2)
                for i, feat in enumerate(feats):
                    with form_cols[i % 2]:
                        default = default_for(feat)
                        inputs[feat] = st.number_input(
                            feat, min_value=0, value=default,
                            step=max(1, default // 100),
                        )
            submitted = st.form_submit_button("Predict tier")

        if submitted:
            template = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    template[col] = float(df[col].median())
                else:
                    template[col] = df[col].mode().iloc[0]
            row = pd.DataFrame([template])
            row["Genre"] = in_genre
            row["Country"] = in_country
            for feat, val in inputs.items():
                row[feat] = val

            for col in SKEWED_COLS:
                log_col = f"{col} Log"
                if log_col in row.columns:
                    row[log_col] = np.log1p(row[col])

            X_input = row.drop(columns=[c for c in DROP_COLS if c in row.columns])
            tier = int(model.predict(X_input)[0])
            probs = model.predict_proba(X_input)[0]
            classes = model.classes_

            # Medal-style badge: gold / silver / bronze for tiers 1 / 2 / 3.
            tier_colors = {1: "#D4AF37", 2: "#9AA3AD", 3: "#B97A40"}
            confidence = float(probs[list(classes).index(tier)])
            st.markdown(
                f"<div style='background:{tier_colors[tier]};color:#1a1a1a;"
                f"padding:16px 22px;border-radius:12px;text-align:center;"
                f"margin:0.4rem 0 0.8rem;'>"
                f"<div style='font-size:1.7rem;font-weight:800;'>"
                f"{TIER_LABELS[tier]}</div>"
                f"<div style='font-size:0.95rem;font-weight:600;'>"
                f"Predicted rank tier · {confidence:.0%} confidence</div></div>",
                unsafe_allow_html=True,
            )

            # Which platform drove this prediction? Combine the model's feature
            # importance with how strong the user's own value is for each metric
            # (its percentile in the data), then roll up to the platform level.
            imp_lookup = dict(zip(pre.get_feature_names_out(), rf.feature_importances_))
            driver_scores = {}
            for feat, val in inputs.items():
                imp = imp_lookup.get(f"num__{feat}", 0.0) + imp_lookup.get(
                    f"num__{feat} Log", 0.0
                )
                percentile = float((df[feat] < val).mean())
                platform = feat.split()[0]
                driver_scores[platform] = driver_scores.get(platform, 0.0) + (
                    percentile * imp
                )
            driver_df = (
                pd.DataFrame(
                    {"Platform": list(driver_scores), "Driver score": list(driver_scores.values())}
                )
                .sort_values("Driver score", ascending=False)
                .reset_index(drop=True)
            )
            top_platform = driver_df.iloc[0]["Platform"]
            st.markdown(
                f"**Biggest driver of this prediction: {top_platform}.** "
                "This combines how influential each platform is to the model with "
                "how strong the values you entered are relative to other artists. "
                "The chart below ranks the platforms pushing this prediction."
            )
            st.bar_chart(driver_df.head(3).set_index("Platform"))

            st.markdown("**Class probabilities**")
            prob_df = pd.DataFrame({
                "Tier": [TIER_LABELS[int(c)] for c in classes],
                "Probability": probs,
            })
            st.bar_chart(prob_df.set_index("Tier"))
            st.caption(
                "Rank tiers are heavily imbalanced (50 / 200 / 2250), so the model "
                "stays conservative about the Top-10 tier even for strong profiles. "
                "The probability bars show the full picture, not just the label."
            )

with about_tab:
    st.header("About this project")
    st.markdown(
        """
        This dashboard analyzes the **Viberate** music artist chart dataset across
        five genres (Pop, Rock, Hip Hop, Electronic, Metal) and predicts each
        artist's per-genre rank tier (Top 10 / Top 11-50 / Top 51+) from their
        cross-platform metrics: Spotify streams and followers, YouTube
        subscribers, TikTok, Instagram, Deezer, and SoundCloud counts.

        **Goal:** identify which platforms actually drive top-tier status.

        **Technical notes:** the dataset is a 2,500-artist snapshot (500 per
        genre). The model is a `RandomForestClassifier` with
        `class_weight="balanced"` inside an sklearn `Pipeline`, trained on ~12 raw
        platform totals plus their `log1p` versions and one-hot–encoded `Genre`
        and `Country`. Performance is reported with **F1-macro** (the right metric
        for the heavy 50 / 200 / 2250 class imbalance), benchmarked against a
        most-frequent-class baseline.

        Built for DS570 — Introduction to Data Science.
        """
    )
