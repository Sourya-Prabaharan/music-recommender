"""Content discovery, with opt-in experimental collaborative ranking."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class MusicRecommender:
    def __init__(self, data_path="data/spotify_sample.csv"):
        self.interactions = pd.read_csv(data_path)
        self.df = self.interactions.drop_duplicates(["artist", "song_name"]).reset_index(drop=True)
        self.item_map = {(row.artist, row.song_name): i for i, row in self.df.iterrows()}
        self.user_map = {}
        self.model = None
        self.features = None
        self.user_item = None

    def build_matrices(self):
        # Scale tempo alongside the 0–1 features; avoid an N-by-N similarity matrix.
        values = self.df[["danceability", "energy", "valence", "tempo"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Audio features must be finite")
        self.features = StandardScaler().fit_transform(values)
        if "user_id" not in self.interactions or self.interactions["user_id"].isna().any():
            return  # A track catalog is not listening history.
        if "play_count" not in self.interactions:
            return
        from scipy.sparse import csr_matrix
        counts = self.interactions["play_count"].to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("Interaction play_count must be finite and positive")
        self.user_map = {u: i for i, u in enumerate(self.interactions["user_id"].unique())}
        rows = self.interactions["user_id"].map(self.user_map)
        cols = [self.item_map[(r.artist, r.song_name)] for r in self.interactions.itertuples()]
        self.user_item = csr_matrix((counts, (rows, cols)), shape=(len(self.user_map), len(self.df)))

    def train_cf(self, factors=32):
        """Experimental: callers must supply genuine user IDs and play counts."""
        if self.user_item is None:
            raise ValueError("Collaborative filtering requires user_id and play_count interactions")
        from implicit.als import AlternatingLeastSquares
        self.model = AlternatingLeastSquares(factors=factors, iterations=20, random_state=0)
        self.model.fit(self.user_item)

    def recommend_cf(self, user_id, n=5):
        if self.model is None:
            raise ValueError("Train the experimental collaborative model with real interactions first")
        uid = self.user_map[user_id]
        ids, scores = self.model.recommend(uid, self.user_item[uid], N=n)
        valid = (ids >= 0) & (ids < len(self.df)) & np.isfinite(scores)
        return self.df.iloc[ids[valid]][["artist", "song_name"]]

    def recommend_content(self, song_name, n=5, artist=None):
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.features is None:
            raise ValueError("Call build_matrices first")
        matches = self.df["song_name"].eq(song_name)
        if artist is not None:
            matches &= self.df["artist"].eq(artist)
        indices = self.df.index[matches]
        if len(indices) != 1:
            raise ValueError("Select one known song; provide artist for ambiguous titles")
        idx = indices[0]
        scores = cosine_similarity(self.features[idx:idx + 1], self.features)[0]
        order = np.argsort(-scores, kind="stable")
        order = order[order != idx][:n]
        return self.df.iloc[order][["artist", "song_name"]]

    def hybrid_recommend(self, user_id, song_name, n=5, alpha=0.6, artist=None):
        """Experimental weighted reciprocal-rank fusion, not a validated model.

        alpha=1 uses only collaborative ranks; alpha=0 uses only content ranks.
        This API is not used by the catalog-only app.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if n < 0:
            raise ValueError("n must be non-negative")
        if alpha == 0:
            return self.recommend_content(song_name, n, artist)
        if alpha == 1:
            return self.recommend_cf(user_id, n)
        scores = {}
        for weight, ranking in (
            (alpha, self.recommend_cf(user_id, len(self.df))),
            (1 - alpha, self.recommend_content(song_name, len(self.df), artist)),
        ):
            for rank, row in enumerate(ranking.itertuples(index=False), 1):
                key = (row.artist, row.song_name)
                scores[key] = scores.get(key, 0) + weight / rank
        selected = sorted(scores, key=scores.get, reverse=True)[:n]
        return pd.DataFrame(selected, columns=["artist", "song_name"])
