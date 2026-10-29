# Spotify Song Recommendation System

A Python content-based music recommender built with the Spotify songs dataset. It suggests similar tracks using song metadata and audio features, and can also return popular or mood-based recommendations.

## Features

- Similar-song recommendations with **TF-IDF** and **cosine similarity**
- Scaled audio features with `MinMaxScaler`
- Popularity-based recommendations
- Mood suggestions using features such as energy, valence, danceability, and acousticness

## Tech stack

Python, pandas, NumPy, scikit-learn, and SciPy.

## Dataset

Save the CSV as `spotify_songs.csv` beside the Python script.

This project uses columns including:

```text
track_id, track_name, track_artist, track_popularity,
playlist_genre, playlist_subgenre, danceability, energy,
loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, duration_ms
```

> **Note:** This CSV uses `track_artist` and `track_popularity`. Update code references to `artist_name` or `popularity` accordingly.

## Installation

```bash
python -m pip install pandas numpy scikit-learn scipy
```

## Usage

```bash
python spotify_recommender.py
```

## Project structure

```text
spotify-song-recommender/
├── spotify_recommender.py
├── spotify_songs.csv
└── README.md
```

## Notes

- This is a content-based system; it does not use listener history.
- Keep the default DataFrame index, or reset it after filtering when using `df.loc`.
- Validate columns before using fallbacks such as `df.columns[1]`.

## License

Add your preferred license here, such as MIT.
