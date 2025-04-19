import pandas as pd
import ast
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# --- Load and preprocess playlist titles ---
df_playlists = pd.read_csv("playlist_titles.csv")
df_playlists['title_clean'] = df_playlists['playlist_title'].fillna("").astype(str).str.lower()

# --- Load and preprocess song titles ---
df_songs = pd.read_csv("song_titles.csv")
df_songs['playlist_titles'] = df_songs['playlist_titles'].apply(ast.literal_eval)

# Optional: tqdm progress for longer playlists
tqdm.pandas(desc="Aggregating titles for songs")
df_songs['all_words'] = df_songs['playlist_titles'].progress_apply(lambda titles: ' '.join(titles).lower())

# --- Fit vectorizer on playlist titles only ---
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(df_playlists['title_clean'])  # FIXED: Pass actual Series, not string

# --- Transform both datasets using shared vocabulary ---
X_playlist = vectorizer.transform(df_playlists['title_clean'])
X_song = vectorizer.transform(df_songs['all_words'])

# --- Save results ---
sp.save_npz("playlist_title_sparse.npz", X_playlist)
sp.save_npz("song_title_sparse.npz", X_song)

pd.Series(df_playlists['playlist_id']).to_csv("playlist_ids.csv", index=False)
pd.Series(df_songs['track_uri']).to_csv("track_uris.csv", index=False)
pd.Series(vectorizer.get_feature_names_out()).to_csv("title_vocab.csv", index=False)

print("✅ Saved sparse matrices and metadata using shared vocabulary.")
