import streamlit as st
import pandas as pd
from music_recommender import MusicRecommender

st.set_page_config(page_title="Music Recommender", page_icon="🎧", layout="wide")
st.title("🎧 Your Personal Music Recommender")
st.write("Find songs similar to your favorites using real Spotify data!")

# ⚡ Cache dataset loading for faster startup
@st.cache_data
def load_recommender():
    rec = MusicRecommender(data_path="data/spotify_sample.csv")
    rec.build_matrices()
    rec.train_cf()
    return rec

with st.spinner("Loading recommender model..."):
    rec = load_recommender()
st.success("Model ready!")

# 🎵 Load song list
song_list = rec.df["song_name"].unique().tolist()

# 🔍 Searchable dropdown
song_choice = st.selectbox("Select a song you like:", song_list, index=None, placeholder="Start typing...")

if song_choice:
    st.subheader(f"🎶 Songs similar to **{song_choice}**:")
    results = rec.recommend_content(song_choice)
    st.dataframe(results, use_container_width=True)

    st.subheader("💡 Hybrid Recommendations:")
    hybrid = rec.hybrid_recommend(user_id=1, song_name=song_choice)
    st.dataframe(hybrid, use_container_width=True)

st.caption("Powered by Spotify Dataset + Streamlit + Python 🎧")

