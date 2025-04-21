import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

# Parameters
PLAYLIST_NAME = "chill vibes"  # Replace or use argparse for dynamic input
TOP_K = 500

# Load data
print("Loading model artifacts...")
U = np.load("Rname_U.npy")          # (num_playlists + num_songs, k)
Vt = np.load("Rname_Vt.npy")        # (k, num_features)
print("Rebuilding vectorizer from playlist_titles.csv...")
playlist_df = pd.read_csv("playlist_titles.csv")
playlist_titles = playlist_df["playlist_title"].astype(str).tolist()

# Refit vectorizer
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(playlist_titles)

# Constants
NUM_PLAYLISTS = 1000000  # Adjust if different

# Step 1: Vectorize the input name
print(f"Encoding playlist name: '{PLAYLIST_NAME}'")
name_vector = vectorizer.transform([PLAYLIST_NAME])  # (1, num_features)

# Step 2: Project into latent space using Vt (like PCA projection)
playlist_embedding = name_vector @ Vt.T  # shape: (1, k)

# Step 3: Get song embeddings (bottom rows of U)
song_embeddings = U[NUM_PLAYLISTS:]  # (num_songs, k)

# Step 4: Compute similarity scores
scores = song_embeddings @ playlist_embedding.T  # (num_songs, 1)
scores = scores.flatten()

# Step 5: Top-K song indices
top_k_indices = scores.argsort()[::-1][:TOP_K]

print(f"\nTop {TOP_K} recommended song indices:")
for idx, song_idx in enumerate(top_k_indices, 1):
    print(f"{idx}. Song row index: {song_idx} | Score: {scores[song_idx]:.4f}")
