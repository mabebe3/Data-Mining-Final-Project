import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Load playlist data
df = pd.read_csv("playlist_titles.csv")

# Clean titles: lowercase and handle missing
df['title_clean'] = df['playlist_title'].fillna("").astype(str).str.lower()

# Create sparse binary bag-of-words matrix
vectorizer = CountVectorizer(binary=True)
X_sparse = vectorizer.fit_transform(df['title_clean'])

# Save sparse matrix
sp.save_npz("playlist_title_sparse.npz", X_sparse)

# Save playlist IDs and vocab
pd.Series(df['playlist_id']).to_csv("playlist_ids.csv", index=False)
pd.Series(vectorizer.get_feature_names_out()).to_csv("playlist_title_vocab.csv", index=False)

print("✅ Saved sparse matrix, playlist IDs, and vocabulary.")
