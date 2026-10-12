from pathlib import Path

import streamlit as st
from music_recommender import MusicRecommender

st.set_page_config(page_title="Music Discovery", page_icon="🎧", layout="wide")
st.title("🎧 Content-based Music Discovery")
st.write("Explore tracks with similar audio features. No listening history or user personalization is used.")


@st.cache_resource
def load_recommender():
    rec = MusicRecommender(Path(__file__).resolve().parents[1] / "data/spotify_sample.csv")
    rec.build_matrices()
    return rec


rec = load_recommender()
choices = list(rec.df[["artist", "song_name"]].itertuples(index=False, name=None))
choice = st.selectbox("Select a track", choices, index=None,
                      format_func=lambda track: f"{track[1]} — {track[0]}")
if choice is not None:
    st.dataframe(rec.recommend_content(choice[1], artist=choice[0]), width="stretch")
st.caption("Similarity uses scaled danceability, energy, valence, and tempo from the included catalog sample.")
