
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Load dataset safely
# -------------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset safely. Auto-detects if file is in the same folder.
    """
    if not os.path.isabs(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = os.path.join(script_dir, csv_path)
        if os.path.exists(candidate_path):
            csv_path = candidate_path

    if not os.path.exists(csv_path):
        print(f"⚠️ Dataset not found at: {csv_path}")
        print("Tip: Place 'spotify_songs.csv' in the same folder as this script "
              "or provide the full path.")
        return pd.DataFrame()

    return pd.read_csv(csv_path, low_memory=False)

# -------------------------------
# Build feature matrix
# -------------------------------
def build_feature_matrix(df: pd.DataFrame):
    text_cols = [c for c in ["track_name", "artist_name", "playlist_genre", "playlist_subgenre"] if c in df.columns]
    num_cols = [c for c in ["popularity","danceability","energy","loudness","speechiness",
                            "acousticness","instrumentalness","liveness","valence","tempo","duration_ms"] if c in df.columns]

    # Text features
    if text_cols:
        text_corpus = df[text_cols].astype(str).agg(" ".join, axis=1)
        tfidf = TfidfVectorizer(stop_words="english")
        X_text = tfidf.fit_transform(text_corpus)
    else:
        from scipy.sparse import csr_matrix
        X_text = csr_matrix((len(df), 1))

    # Numeric features
    if num_cols:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(df[num_cols].fillna(0))
    else:
        X_num = np.zeros((len(df), 1))

    # Combine
    from scipy.sparse import hstack, csr_matrix
    X = hstack([X_text, csr_matrix(X_num)]).tocsr()

    meta = {
        "text_cols": text_cols,
        "num_cols": num_cols,
        "title_col": "track_name" if "track_name" in df.columns else df.columns[0],
        "artist_col": "artist_name" if "artist_name" in df.columns else df.columns[1]
    }
    return X, meta

# -------------------------------
# Recommendation engine
# -------------------------------
def recommend_similar(df, X, meta, query_title, query_artist=None, top_k=10):
    title_col = meta["title_col"]
    artist_col = meta["artist_col"]

    # Find index
    idx = None
    for i, row in df.iterrows():
        if str(row[title_col]).lower() == query_title.lower():
            if query_artist:
                if str(row[artist_col]).lower() == query_artist.lower():
                    idx = i
                    break
            else:
                idx = i
                break

    if idx is None:
        raise ValueError("Song not found in dataset.")

    sims = cosine_similarity(X[idx], X).flatten()
    sims[idx] = -1
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = df.loc[top_indices, [title_col, artist_col]].copy()
    results.insert(0, "similarity", sims[top_indices])
    return results.reset_index(drop=True)

# -------------------------------
# Baseline recommender
# -------------------------------
def baseline_popularity(df, meta, top_k=10):
    if "popularity" in meta["num_cols"]:
        top = df.sort_values("popularity", ascending=False)
    else:
        top = df.copy()
    return top[[meta["title_col"], meta["artist_col"]]].head(top_k).reset_index(drop=True)

# -------------------------------
# Mood-based recommender
# -------------------------------
def recommend_by_mood(df, meta, mood, top_k=10):
    """
    Recommend songs based on user mood using audio features.
    """
    title_col = meta["title_col"]
    artist_col = meta["artist_col"]

    mood_map = {
        "happy": ("valence", True),       # high valence
        "sad": ("valence", False),        # low valence
        "energetic": ("energy", True),    # high energy
        "calm": ("acousticness", True),   # high acousticness
        "party": ("danceability", True),  # high danceability
    }

    if mood not in mood_map:
        print(f"⚠️ Mood '{mood}' not recognized. Try: happy, sad, energetic, calm, party.")
        return pd.DataFrame()

    feature, high = mood_map[mood]
    if feature not in meta["num_cols"]:
        print(f"⚠️ Feature '{feature}' not found in dataset.")
        return pd.DataFrame()

    if high:
        recs = df.sort_values(feature, ascending=False).head(top_k)
    else:
        recs = df.sort_values(feature, ascending=True).head(top_k)

    return recs[[title_col, artist_col, feature]].reset_index(drop=True)

# -------------------------------
# Demo runner
# -------------------------------
def build_and_demo(csv_path="spotify_songs.csv", sample_title=None, sample_artist=None, top_k=10):
    df = load_dataset(csv_path)
    if df.empty:
        print("❌ No dataset loaded. Please check your file path.")
        return

    X, meta = build_feature_matrix(df)

    print("✅ Dataset loaded successfully")
    print(f"Rows: {len(df)}")
    print(f"Text columns: {meta['text_cols']}")
    print(f"Numeric columns: {meta['num_cols']}")
    print("-" * 50)

    print("🎵 Baseline (Top popular tracks):")
    print(baseline_popularity(df, meta, top_k=10))
    print("-" * 50)

    if sample_title:
        try:
            recs = recommend_similar(df, X, meta, sample_title, sample_artist, top_k=top_k)
            print(f"Query track: {sample_title} — {sample_artist if sample_artist else ''}")
            print("\nTop recommendations:")
            print(recs)
        except ValueError as e:
            print("Recommendation error:", e)

    # 🔥 Mood-based recommendation
    user_mood = input("Enter your mood (happy, sad, energetic, calm, party): ").strip().lower()
    mood_recs = recommend_by_mood(df, meta, user_mood, top_k=10)
    if not mood_recs.empty:
        print(f"\n🎶 Songs recommended for mood '{user_mood}':")
        print(mood_recs)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    csv_path = "spotify_songs.csv"  # auto-detects in same folder
    # Uncomment if you prefer absolute path:
    # csv_path = r"c:\Users\harsh\Desktop\python\song recomd sys\spotify_songs.csv"

    build_and_demo(
        csv_path=csv_path,
        sample_title="Shape of You",   # change to a song in your dataset
        sample_artist="Ed Sheeran",    # change to an artist in your dataset
        top_k=10
    )