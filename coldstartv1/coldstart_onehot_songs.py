import pandas as pd
import ast
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the dataset
df = pd.read_csv("song_titles.csv")

# Convert the playlist titles from strings to actual lists
df['playlist_titles'] = df['playlist_titles'].apply(ast.literal_eval)

# Combine the list of titles into one lowercase string
df['all_words'] = df['playlist_titles'].apply(lambda titles: ' '.join(titles).lower())

# Create sparse binary feature matrix
vectorizer = CountVectorizer(binary=True)
X_sparse = vectorizer.fit_transform(df['all_words'])  # this stays sparse

# Save the sparse matrix and the metadata (track_uri and vocab)
sp.save_npz("song_title_sparse.npz", X_sparse)

# Save track URIs and feature names separately
pd.Series(df['track_uri']).to_csv("track_uris.csv", index=False)
pd.Series(vectorizer.get_feature_names_out()).to_csv("title_vocab.csv", index=False)

print("✅ Saved sparse matrix, track URIs, and vocab.")
