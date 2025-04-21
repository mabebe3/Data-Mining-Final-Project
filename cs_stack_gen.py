# generate_sparse_matrix.py

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack, save_npz

# Input files
SONG_CSV = "song_titles.csv"
PLAYLIST_CSV = "playlist_titles.csv"

# Load CSVs
playlist_df = pd.read_csv(PLAYLIST_CSV)
song_df = pd.read_csv(SONG_CSV)

# Extract titles
playlist_titles = playlist_df["playlist_title"].astype(str).tolist()

# Convert stringified lists back to Python lists for songs
song_titles_lists = song_df["playlist_titles"].apply(ast.literal_eval)
song_titles_joined = [' '.join(set(titles)) for titles in song_titles_lists]

# Vectorize
vectorizer = CountVectorizer(binary=True)
playlist_matrix = vectorizer.fit_transform(playlist_titles)
song_matrix = vectorizer.transform(song_titles_joined)

# Stack matrices
Rname_sparse = vstack([playlist_matrix, song_matrix])
print(f"Rname matrix shape: {Rname_sparse.shape}")

# Save sparse matrix
save_npz("Rname_sparse.npz", Rname_sparse)
print("Sparse matrix saved as 'Rname_sparse.npz'")
