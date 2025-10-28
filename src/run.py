from music_recommender import MusicRecommender

# use your Spotify data file
rec = MusicRecommender(data_path="data/spotify_tracks.csv")
rec.build_matrices()
rec.train_cf()

print("🎵 Collaborative-filtering recommendations:")
print(rec.recommend_cf(user_id=1))

print("\n🎶 Content-based recommendations:")
print(rec.recommend_content("Shameless"))

print("\n💡 Hybrid recommendations:")
print(rec.hybrid_recommend(user_id=1, song_name="Shameless"))

