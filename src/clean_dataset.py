import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/dataset 2.csv")

# Rename columns to fit the recommender format
df = df.rename(columns={
    "track_name": "song_name",
    "artists": "artist"
})

# Keep only the most relevant columns
columns_to_keep = ["song_name", "artist", "danceability", "energy", "valence", "tempo"]
df = df[columns_to_keep]

# Clean up missing values and duplicates
df = df.dropna().drop_duplicates(subset=["song_name", "artist"])

# Save as the new dataset for the recommender
df.to_csv("data/spotify_large.csv", index=False)
print(f"✅ Cleaned dataset saved! {len(df)} tracks → data/spotify_large.csv")

