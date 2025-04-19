import pandas as pd
import numpy as np
from scipy.sparse import load_npz, vstack
from sklearn.decomposition import TruncatedSVD

# Load matrices
X_playlist = load_npz("playlist_title_sparse.npz")  # shape: (num_playlists, num_words)
X_song = load_npz("song_title_sparse.npz")          # shape: (num_songs, num_words)

# Stack into Rname
Rname = vstack([X_playlist, X_song]).tocsr()

# Run SVD
svd_dim = 64
print(f"Running Truncated SVD with {svd_dim} components...")
svd = TruncatedSVD(n_components=svd_dim)
Uname = svd.fit_transform(Rname)  # (num_playlists + num_songs, svd_dim)
Vname = svd.components_           # (svd_dim, num_words)

# Split the embedding back
Uname_playlists = Uname[:X_playlist.shape[0]]
Uname_songs = Uname[X_playlist.shape[0]:]

# Save
np.save("svd_playlist_name_vectors.npy", Uname_playlists)
np.save("svd_song_name_vectors.npy", Uname_songs)
np.save("svd_word_matrix_Vname.npy", Vname)

print("✅ SVD complete. Saved latent vectors for playlists and songs.")
