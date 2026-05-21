import plotly.express as px
import streamlit as st

from music.process import load_processed

st.set_page_config(
    page_title="Artist Rank Predictor",
    page_icon=":musical_note:",
    layout="wide",
)

st.title("Artist Rank Predictor")
st.caption(
    "Predicting artist rank tier from cross-platform music metrics — DS570 Final Project"
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

with ml_tab:
    st.header("Model Results")
    st.info("Model evaluation and feature importance coming next.")

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

        Built for DS570 — Introduction to Data Science.
        """
    )
