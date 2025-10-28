import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-top-read"
))

def fetch_user_tracks(limit=20):
    """Fetch user's top tracks without calling the restricted audio-features endpoint."""
    results = sp.current_user_top_tracks(limit=limit, time_range="medium_term")
    songs = []
    for item in results["items"]:
        songs.append({
            "song_name": item["name"],
            "artist": item["artists"][0]["name"],
            "album": item["album"]["name"],
            "popularity": item["popularity"],
            # basic placeholders so your recommender still runs
            "danceability": item["popularity"] / 100,
            "energy": 0.5,
            "valence": 0.5,
            "tempo": 100
        })

    df = pd.DataFrame(songs)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/spotify_tracks.csv", index=False)
    print(f"✅ Saved {len(df)} tracks (basic features) to data/spotify_tracks.csv")
    return df

if __name__ == "__main__":
    fetch_user_tracks()


