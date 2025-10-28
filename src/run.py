from music_recommender import MusicRecommender

rec = MusicRecommender()
rec.build_matrices()
rec.train_cf()

print("CF recommendations:")
print(rec.recommend_cf(user_id=1))

print("\nContent recommendations:")
print(rec.recommend_content("Viva La Vida"))

print("\nHybrid recommendations:")
print(rec.hybrid_recommend(user_id=1, song_name="Viva La Vida"))

