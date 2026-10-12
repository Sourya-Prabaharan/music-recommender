"""Run a catalog-only example without training collaborative filtering."""
from pathlib import Path
from music_recommender import MusicRecommender

rec = MusicRecommender(Path(__file__).resolve().parents[1] / "data/spotify_sample.csv")
rec.build_matrices()
seed = rec.df.iloc[0]
print(f"Tracks similar to {seed.song_name} by {seed.artist}:")
print(rec.recommend_content(seed.song_name, artist=seed.artist).to_string(index=False))
