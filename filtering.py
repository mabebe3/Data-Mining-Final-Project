import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import faiss
import time

# Load data
playlist_tracks_df = pd.read_csv('data/playlists_tracks.csv')
playlists_df = pd.read_csv('data/playlists.csv')
tracks_df = pd.read_csv('data/tracks.csv')
artists_df = pd.read_csv('data/artists.csv')

# Create mappings
unique_playlists = playlist_tracks_df['playlist_id'].unique()
unique_tracks = playlist_tracks_df['track_id'].unique()
playlist_to_idx = {pid: i for i, pid in enumerate(unique_playlists)}
track_to_idx = {tid: i for i, tid in enumerate(unique_tracks)}
idx_to_playlist = {i: pid for pid, i in playlist_to_idx.items()}
idx_to_track = {i: tid for tid, i in track_to_idx.items()}

# Build interaction matrix
rows = [track_to_idx[tid] for tid in playlist_tracks_df['track_id']]
cols = [playlist_to_idx[pid] for pid in playlist_tracks_df['playlist_id']]
data = [1.0] * len(rows)  # must be floats for SVD
interaction_matrix = sp.csr_matrix((data, (rows, cols)), 
                                  shape=(len(unique_tracks), len(unique_playlists)),
                                  dtype=np.float32)

print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Matrix density: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.6f}")

# Create improved inverted indices for quick lookups
track_to_playlists = {}  # Maps track_idx -> [playlist_idx1, playlist_idx2, ...]
playlist_to_tracks = {}  # Maps playlist_idx -> [track_idx1, track_idx2, ...]

coo_matrix = interaction_matrix.tocoo()
for i, j, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
    # i = track_idx, j = playlist_idx
    if i not in track_to_playlists:
        track_to_playlists[i] = []
    track_to_playlists[i].append(j)
    
    if j not in playlist_to_tracks:
        playlist_to_tracks[j] = []
    playlist_to_tracks[j].append(i)

# Perform SVD
k = 100  # number of latent features
U, sigma, Vt = svds(interaction_matrix, k=k)
idx = np.argsort(-sigma)  # sorted in descending order
sigma = sigma[idx]
U = U[:, idx]
Vt = Vt[idx, :]
sigma_diag = np.diag(sigma)
track_factors = U.dot(np.sqrt(sigma_diag))  # latent embedding
playlist_factors = np.sqrt(sigma_diag).dot(Vt).T  # latent embedding

# Build FAISS index for approximate nearest neighbors
# Convert to float32 (required by FAISS)
track_vectors = track_factors.astype('float32')
dimension = track_vectors.shape[1]  # number of latent factors

# Use IndexFlatIP for inner product similarity (cosine without normalization)
# For cosine similarity, we should normalize the vectors first
norms = np.linalg.norm(track_vectors, axis=1, keepdims=True)
normalized_vectors = track_vectors / norms
index = faiss.IndexFlatIP(dimension)
index.add(normalized_vectors)

# For larger datasets, use approximate search with IndexIVFFlat
# This creates clusters (nlist=100) for faster search
nlist = min(100, len(track_vectors) // 39)  # Rule of thumb: sqrt(n)
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(normalized_vectors)
index_ivf.add(normalized_vectors)
# Number of clusters to explore during search (higher = more accurate but slower)
index_ivf.nprobe = 10  

# Create track and playlist dataframes with metadata
track_factors_df = pd.DataFrame(track_factors, columns=[f'factor_{i}' for i in range(k)])
track_factors_df['track_id'] = [idx_to_track[i] for i in range(len(track_factors_df))]
track_factors_with_meta = track_factors_df.merge(tracks_df[['track_id', 'track_name', 'artist_id']], on='track_id')
track_factors_with_meta = track_factors_with_meta.merge(artists_df[['artist_id', 'artist_name']], on='artist_id')

playlist_factors_df = pd.DataFrame(playlist_factors, columns=[f'factor_{i}' for i in range(k)])
playlist_factors_df['playlist_id'] = [idx_to_playlist[i] for i in range(len(playlist_factors_df))]
playlist_factors_with_meta = playlist_factors_df.merge(playlists_df[['playlist_id', 'name', 'num_tracks']], on='playlist_id')

# Save if needed
# track_factors_with_meta.to_csv('track_latent_factors.csv', index=False)
# playlist_factors_with_meta.to_csv('playlist_latent_factors.csv', index=False)

def get_track_name(track_idx):
    """Get track name from track index"""
    track_id = idx_to_track[track_idx]
    track_info = tracks_df[tracks_df['track_id'] == track_id]
    if not track_info.empty:
        return track_info['track_name'].values[0]
    return f"Unknown Track (ID: {track_id})"

def get_co_occurrence_recommendations(track_idx, top_n=10):
    """Get recommendations based on track co-occurrences in playlists"""
    # Find all playlists containing this track
    playlists = track_to_playlists.get(track_idx, [])
    
    # Count co-occurring tracks
    co_occurrence_counts = {}
    for playlist_idx in playlists:
        for co_track_idx in playlist_to_tracks[playlist_idx]:
            if co_track_idx != track_idx:
                co_occurrence_counts[co_track_idx] = co_occurrence_counts.get(co_track_idx, 0) + 1
    
    # Sort by count and normalize
    if co_occurrence_counts:
        max_count = max(co_occurrence_counts.values())
        recommended_tracks = [(idx, count/max_count) for idx, count in co_occurrence_counts.items()]
        recommended_tracks = sorted(recommended_tracks, key=lambda x: x[1], reverse=True)[:top_n]
        return recommended_tracks
    return []

def get_ann_recommendations(track_idx, top_n=10):
    """Get recommendations using FAISS ANN search"""
    query_vector = normalized_vectors[track_idx:track_idx+1]
    distances, indices = index_ivf.search(query_vector, top_n + 1)  # +1 because it will include the query itself
    
    # Filter out the query track and normalize scores to 0-1 range
    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx != track_idx and idx < len(idx_to_track):  # Valid index check
            # Convert distance to similarity score (0-1 range)
            # For inner product, higher is better and max is 1.0 for normalized vectors
            results.append((idx, float(dist)))
    
    # Sort by distance (higher is better for inner product similarity)
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    return results

def hybrid_recommendations(track_idx, alpha=0.5, top_n=10):
    """Combine ANN and co-occurrence recommendations with weighting factor alpha"""
    # Get ANN recommendations
    ann_start = time.time()
    ann_recs = get_ann_recommendations(track_idx, top_n*2)
    ann_recs_dict = {idx: score for idx, score in ann_recs}
    print(f"ANN search time: {time.time() - ann_start:.4f} seconds")
    
    # Get co-occurrence recommendations
    co_start = time.time()
    co_occur_recs = get_co_occurrence_recommendations(track_idx, top_n*2)
    co_occur_dict = {idx: score for idx, score in co_occur_recs}
    print(f"Co-occurrence calculation time: {time.time() - co_start:.4f} seconds")
    
    # Combine scores
    combined_scores = {}
    all_tracks = set(list(ann_recs_dict.keys()) + list(co_occur_dict.keys()))
    
    for t in all_tracks:
        ann_score = ann_recs_dict.get(t, 0)
        co_score = co_occur_dict.get(t, 0)
        combined_scores[t] = alpha * ann_score + (1-alpha) * co_score
    
    # Return top recommendations
    top_tracks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(idx, score) for idx, score in top_tracks]

def recommend_tracks_for_playlist(playlist_idx, alpha=0.5, top_n=10):
    """Recommend tracks for a playlist using hybrid approach"""
    # Get current tracks in playlist
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    
    # Get playlist vector
    playlist_vector = playlist_factors[playlist_idx].reshape(1, -1)
    normalized_playlist = playlist_vector / np.linalg.norm(playlist_vector)
    
    # Find similar tracks using ANN
    distances, indices = index_ivf.search(normalized_playlist.astype('float32'), top_n*3)
    ann_recs = {idx: float(dist) for idx, dist in zip(indices[0], distances[0]) if idx not in current_tracks}
    
    # Find tracks from similar playlists (collaborative filtering approach)
    similar_playlists = {}
    for track_idx in current_tracks:
        for p_idx in track_to_playlists[track_idx]:
            if p_idx != playlist_idx:
                similar_playlists[p_idx] = similar_playlists.get(p_idx, 0) + 1
    
    # Get candidate tracks from similar playlists
    co_occur_dict = {}
    for p_idx, overlap in sorted(similar_playlists.items(), key=lambda x: x[1], reverse=True)[:20]:
        for t_idx in playlist_to_tracks[p_idx]:
            if t_idx not in current_tracks:
                co_occur_dict[t_idx] = co_occur_dict.get(t_idx, 0) + overlap
    
    # Normalize co-occurrence scores
    if co_occur_dict:
        max_co = max(co_occur_dict.values())
        co_occur_dict = {k: v/max_co for k, v in co_occur_dict.items()}
    
    # Combine scores
    combined_scores = {}
    all_tracks = set(list(ann_recs.keys()) + list(co_occur_dict.keys()))
    
    for t in all_tracks:
        ann_score = ann_recs.get(t, 0)
        co_score = co_occur_dict.get(t, 0)
        combined_scores[t] = alpha * ann_score + (1-alpha) * co_score
    
    # Return top recommendations
    top_tracks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(idx, score) for idx, score in top_tracks]

# Example usage
if __name__ == "__main__":
    # Example track to use for recommendations
    track_idx = 123
    print(f"\nGenerating recommendations for track: {get_track_name(track_idx)}")
    
    # Get pure ANN recommendations
    print("\nPure ANN Recommendations:")
    ann_recs = get_ann_recommendations(track_idx)
    for i, (idx, score) in enumerate(ann_recs):
        print(f"Rank {i+1}: {get_track_name(idx)} (Score: {score:.4f})")
    
    # Get pure co-occurrence recommendations
    print("\nPure Co-occurrence Recommendations:")
    co_recs = get_co_occurrence_recommendations(track_idx)
    for i, (idx, score) in enumerate(co_recs):
        print(f"Rank {i+1}: {get_track_name(idx)} (Score: {score:.4f})")
    
    # Get hybrid recommendations
    print("\nHybrid Recommendations (Alpha=0.7):")
    hybrid_recs = hybrid_recommendations(track_idx, alpha=0.7)
    for i, (idx, score) in enumerate(hybrid_recs):
        print(f"Rank {i+1}: {get_track_name(idx)} (Score: {score:.4f})")
    
    # Example playlist recommendation
    playlist_idx = 50  # Replace with a valid playlist index
    playlist_id = idx_to_playlist[playlist_idx]
    playlist_name = playlists_df[playlists_df['playlist_id'] == playlist_id]['name'].values[0]
    
    print(f"\nRecommending tracks for playlist: {playlist_name}")
    playlist_recs = recommend_tracks_for_playlist(playlist_idx)
    for i, (idx, score) in enumerate(playlist_recs):
        print(f"Rank {i+1}: {get_track_name(idx)} (Score: {score:.4f})")