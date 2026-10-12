# Content-based Music Discovery

A Python/Streamlit app that finds similar tracks using danceability, energy, valence, and tempo. The working app uses the included 2,000-row catalog sample. It does not learn a user's preferences or use listening history.

## Run locally

Use Python 3.11+ from the repository root:

```bash
git clone https://github.com/Sourya-Prabaharan/music-recommender.git
cd music-recommender
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run src/app.py
```

Open http://localhost:8501. For a terminal example: `python src/run.py`.

The previously advertised `music-recommender.streamlit.app` URL could not be verified (redirect loop during the audit). Use the local app; this README does not claim a working hosted deployment.

## Implementation

[MusicRecommender](src/music_recommender.py) scales audio features so tempo does not dominate solely because it uses larger units, computes cosine similarity on demand, and excludes the selected track. Artist/title pairs distinguish tracks with the same title. It stores O(N × features) data rather than a dense O(N²) similarity matrix.

The larger catalog CSV has 114,000 rows; the app uses `data/spotify_sample.csv`, not the full catalog. The CSVs contain track metadata, not validated listening histories. The five-row `songs.csv` example has user IDs/play counts but no provenance establishing genuine behavioral data.

To use the optional ALS dependency, install `requirements-experimental.txt`.

Collaborative ALS and weighted reciprocal-rank fusion remain **experimental APIs** for separately supplied genuine interactions. They require explicit `user_id` and positive `play_count`; the app does not invoke them. `alpha` controls collaborative versus content rank weights. No personalization quality or hybrid evaluation result is claimed.

## Checks and limitations

```bash
python -m pip install pytest
python -m pytest -q
python src/run.py
```

Tests cover catalog-only behavior, exclusion of the seed, ambiguous titles, interaction counts, and ranking weights. Similarity is an exploratory audio-feature heuristic, not a measured recommendation-quality result. Dataset licensing/redistribution terms should be reviewed before reuse.
