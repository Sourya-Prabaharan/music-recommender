import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self, data_path="data/songs.csv"):
        self.df = pd.read_csv(data_path)
        self.user_map = {u:i for i,u in enumerate(self.df['user_id'].unique())}
        self.item_map = {s:i for i,s in enumerate(self.df['song_id'].unique())}
        self.inv_user = {v:k for k,v in self.user_map.items()}
        self.inv_item = {v:k for k,v in self.item_map.items()}
        self.model = None
        self.content_sim = None

    def build_matrices(self):
        rows = self.df['user_id'].map(self.user_map)
        cols = self.df['song_id'].map(self.item_map)
        data = self.df['play_count']
        self.user_item = sp.csr_matrix((data, (rows, cols)))

        features = self.df[['danceability','energy','valence','tempo']].values
        self.content_sim = cosine_similarity(features)

    def train_cf(self, factors=32):
        model = AlternatingLeastSquares(factors=factors, iterations=20)
        model.fit(self.user_item)
        self.model = model

    def recommend_cf(self, user_id, n=5):
        uid = self.user_map[user_id]
        ids, scores = self.model.recommend(uid, self.user_item[uid], N=n)
        recs = self.df[self.df['song_id'].isin([self.inv_item[i] for i in ids])]
        return recs[['artist','song_name']]

    def recommend_content(self, song_name, n=5):
        idx = self.df[self.df['song_name'] == song_name].index[0]
        sims = np.argsort(-self.content_sim[idx])[:n]
        return self.df.iloc[sims][['artist','song_name']]

    def hybrid_recommend(self, user_id, song_name, n=5, alpha=0.6):
        cf_recs = self.recommend_cf(user_id, n)
        content_recs = self.recommend_content(song_name, n)
        combined = pd.concat([cf_recs, content_recs]).drop_duplicates()
        return combined.head(n)

