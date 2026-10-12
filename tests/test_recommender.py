import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommender import MusicRecommender


@pytest.fixture
def rec(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({"artist": ["A", "B", "C", "D"], "song_name": ["seed", "near", "far", "far"],
                  "danceability": [.1, .11, .9, .8], "energy": [.2, .21, .8, .7],
                  "valence": [.3, .31, .7, .6], "tempo": [100, 101, 150, 140]}).to_csv(path, index=False)
    result = MusicRecommender(path)
    result.build_matrices()
    return result


def test_catalog_does_not_create_users_or_dense_pairwise_matrix(rec):
    assert "user_id" not in rec.df
    assert rec.user_item is None
    assert rec.features.shape == (4, 4)
    with pytest.raises(ValueError, match="requires user_id"):
        rec.train_cf()


def test_content_excludes_seed_and_preserves_artist_identity(rec):
    rows = rec.recommend_content("seed", 10)
    assert rows.iloc[0].song_name == "near"
    assert len(rows) == 3
    assert list(rows.song_name).count("far") == 2
    assert rec.recommend_content("seed", 0).empty


def test_unknown_and_ambiguous_titles(rec):
    for name in ["missing", "far"]:
        with pytest.raises(ValueError, match="Select one"):
            rec.recommend_content(name)
    assert len(rec.recommend_content("far", artist="C")) == 3


def test_hybrid_weight_changes_ranking_without_concatenation_bias(rec):
    rec.recommend_cf = Mock(return_value=rec.df.iloc[[2, 1]][["artist", "song_name"]])
    assert rec.hybrid_recommend(1, "seed", 1, alpha=.9).iloc[0].song_name == "far"
    assert rec.hybrid_recommend(1, "seed", 1, alpha=.1).iloc[0].song_name == "near"
    rec.recommend_cf.reset_mock()
    assert rec.hybrid_recommend(1, "seed", 1, alpha=0).iloc[0].song_name == "near"
    rec.recommend_cf.assert_not_called()
    with pytest.raises(ValueError, match="alpha"):
        rec.hybrid_recommend(1, "seed", alpha=1.1)


def test_interactions_preserve_counts_and_distinct_artists(tmp_path):
    path = tmp_path / "listens.csv"
    pd.DataFrame({"artist": ["A", "B", "A"], "song_name": ["same"] * 3,
                  "user_id": [5, 5, 6], "play_count": [2, 3, 4],
                  "danceability": [.1, .9, .1], "energy": [.2, .8, .2],
                  "valence": [.3, .7, .3], "tempo": [100, 140, 100]}).to_csv(path, index=False)
    rec = MusicRecommender(path)
    rec.build_matrices()
    assert rec.user_item.toarray().tolist() == [[2, 3], [4, 0]]


def test_collaborative_filters_missing_candidates(rec):
    import numpy as np
    from scipy.sparse import csr_matrix
    rec.user_map = {9: 0}
    rec.user_item = csr_matrix([[1, 0, 0, 0]])
    rec.model = Mock()
    rec.model.recommend.return_value = (np.array([1, -1, 3]), np.array([.9, -np.inf, -np.inf]))
    assert rec.recommend_cf(9).song_name.tolist() == ["near"]


def test_catalog_cannot_run_hybrid_personalization(rec):
    with pytest.raises(ValueError, match="real interactions"):
        rec.hybrid_recommend(1, "seed")
