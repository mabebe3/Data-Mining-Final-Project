import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import faiss
import time
import os

# # Load training data
# playlist_tracks_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\data_formatted\\playlists_tracks.csv')
# playlists_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\data_formatted\\playlists.csv')
# tracks_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\data_formatted\\tracks.csv')
# artists_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\data_formatted\\artists.csv')

# # Load challenge data
# challenge_playlist_tracks_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\online\\playlists_tracks.csv')
# challenge_playlists_df = pd.read_csv('C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\online\\playlists.csv')

challenge_playlists_df = challenge_playlists_df[challenge_playlists_df['num_samples'] > 0]

print(f"Training playlists: {len(playlists_df)}")
print(f"Challenge playlists: {len(challenge_playlists_df)}")

# Process both training and challenge playlists
all_playlist_tracks = pd.concat([playlist_tracks_df, challenge_playlist_tracks_df])
all_playlists = pd.concat([playlists_df, challenge_playlists_df])

# Create unified mappings for all playlists and tracks
unique_playlists = all_playlist_tracks['playlist_id'].unique()
unique_tracks = all_playlist_tracks['track_id'].unique()
playlist_to_idx = {pid: i for i, pid in enumerate(unique_playlists)}
track_to_idx = {tid: i for i, tid in enumerate(unique_tracks)}
idx_to_playlist = {i: pid for pid, i in playlist_to_idx.items()}
idx_to_track = {i: tid for tid, i in track_to_idx.items()}

print(f"Total unique playlists: {len(unique_playlists)}")
print(f"Total unique tracks: {len(unique_tracks)}")

# Build the interaction matrix using only training data
rows = [track_to_idx[tid] for tid in playlist_tracks_df['track_id']]
cols = [playlist_to_idx[pid] for pid in playlist_tracks_df['playlist_id']]
data = [1.0] * len(rows)  # must be floats for SVD
interaction_matrix = sp.csr_matrix((data, (rows, cols)), 
                                  shape=(len(unique_tracks), len(unique_playlists)),
                                  dtype=np.float32)

print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Matrix density: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.6f}")

# Create inverted indices for quick lookups
track_to_playlists = {}  # Maps track_idx -> [playlist_idx1, playlist_idx2, ...]
playlist_to_tracks = {}  # Maps playlist_idx -> [track_idx1, track_idx2, ...]

# Process training data for inverted indices
coo_matrix = interaction_matrix.tocoo()
for i, j, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
    # i = track_idx, j = playlist_idx
    if i not in track_to_playlists:
        track_to_playlists[i] = []
    track_to_playlists[i].append(j)
    
    if j not in playlist_to_tracks:
        playlist_to_tracks[j] = []
    playlist_to_tracks[j].append(i)

# Process challenge data for inverted indices
for _, row in challenge_playlist_tracks_df.iterrows():
    playlist_idx = playlist_to_idx[row['playlist_id']]
    track_idx = track_to_idx[row['track_id']]
    
    if track_idx not in track_to_playlists:
        track_to_playlists[track_idx] = []
    track_to_playlists[track_idx].append(playlist_idx)
    
    if playlist_idx not in playlist_to_tracks:
        playlist_to_tracks[playlist_idx] = []
    playlist_to_tracks[playlist_idx].append(track_idx)

# Perform SVD on the training data
k = 100  # number of latent features
print("Performing SVD...")
U, sigma, Vt = svds(interaction_matrix, k=k)
idx = np.argsort(-sigma)  # sorted in descending order
sigma = sigma[idx]
U = U[:, idx]
Vt = Vt[idx, :]
sigma_diag = np.diag(sigma)
track_factors = U.dot(np.sqrt(sigma_diag))  # latent embedding
playlist_factors = np.sqrt(sigma_diag).dot(Vt).T  # latent embedding
print("SVD completed")

# Function to calculate playlist factors for challenge playlists
def calculate_playlist_factor(playlist_idx):
    """Calculate playlist factor for a playlist using its tracks' factors"""
    track_indices = playlist_to_tracks.get(playlist_idx, [])
    if not track_indices:
        return np.zeros(k)
    
    # Average the track factors
    track_vecs = [track_factors[idx] for idx in track_indices if idx < len(track_factors)]
    if not track_vecs:
        return np.zeros(k)
    
    return np.mean(track_vecs, axis=0)

# Calculate playlist factors for challenge playlists
challenge_playlist_factors = {}
for pid in challenge_playlists_df['playlist_id'].unique():
    playlist_idx = playlist_to_idx[pid]
    challenge_playlist_factors[playlist_idx] = calculate_playlist_factor(playlist_idx)

# Add challenge playlist factors to playlist_factors
for playlist_idx, factor in challenge_playlist_factors.items():
    if playlist_idx >= len(playlist_factors):
        playlist_factors = np.vstack([playlist_factors, factor.reshape(1, -1)])

# Build FAISS index for tracks
print("Building FAISS index...")
track_vectors = track_factors.astype('float32')
dimension = track_vectors.shape[1]  # number of latent factors

# For cosine similarity, normalize the vectors
norms = np.linalg.norm(track_vectors, axis=1, keepdims=True)
normalized_vectors = track_vectors / norms

# Create approximate nearest neighbors index
nlist = min(100, len(track_vectors) // 39)  # Rule of thumb: sqrt(n)
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(normalized_vectors)
index_ivf.add(normalized_vectors)
index_ivf.nprobe = 20  # Number of clusters to explore during search
print("FAISS index built")

def get_track_name(track_idx):
    """Get track name from track index"""
    track_id = idx_to_track[track_idx]
    track_info = tracks_df[tracks_df['track_id'] == track_id]
    if not track_info.empty:
        return track_info['track_name'].values[0]
    return f"Unknown Track (ID: {track_id})"

def get_co_occurrence_recommendations(playlist_idx, top_n=500):
    """Get recommendations based on co-occurrences for a playlist"""
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    if not current_tracks:
        return []
    
    # Find similar playlists (those sharing at least one track)
    similar_playlists = {}
    for track_idx in current_tracks:
        for p_idx in track_to_playlists.get(track_idx, []):
            if p_idx != playlist_idx:
                similar_playlists[p_idx] = similar_playlists.get(p_idx, 0) + 1
    
    # Get candidate tracks from similar playlists
    co_occur_dict = {}
    for p_idx, overlap in sorted(similar_playlists.items(), key=lambda x: x[1], reverse=True)[:30]:
        for t_idx in playlist_to_tracks.get(p_idx, []):
            if t_idx not in current_tracks:
                co_occur_dict[t_idx] = co_occur_dict.get(t_idx, 0) + overlap
    
    # Normalize co-occurrence scores
    if co_occur_dict:
        max_co = max(co_occur_dict.values())
        recommended_tracks = [(idx, count/max_co) for idx, count in co_occur_dict.items()]
        recommended_tracks = sorted(recommended_tracks, key=lambda x: x[1], reverse=True)[:top_n]
        return recommended_tracks
    return []

def get_ann_recommendations(playlist_idx, top_n=500):
    """Get recommendations using FAISS ANN search for a playlist"""
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    
    # Get playlist vector
    if playlist_idx < len(playlist_factors):
        playlist_vector = playlist_factors[playlist_idx].reshape(1, -1)
    else:
        playlist_vector = challenge_playlist_factors[playlist_idx].reshape(1, -1)
    
    # Normalize the vector
    norm = np.linalg.norm(playlist_vector)
    if norm > 0:
        normalized_playlist = playlist_vector / norm
    else:
        return []
    
    # Search for similar tracks
    distances, indices = index_ivf.search(normalized_playlist.astype('float32'), top_n + len(current_tracks))
    
    # Filter out tracks already in the playlist
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(idx_to_track) and idx not in current_tracks:
            results.append((idx, float(dist)))
    
    # Sort by similarity
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    return results

def hybrid_recommendations(playlist_idx, alpha=0.7, top_n=500):
    """Combine ANN and co-occurrence recommendations with weighting factor alpha"""
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    
    # Get ANN recommendations
    ann_recs = get_ann_recommendations(playlist_idx, top_n*2)
    ann_recs_dict = {idx: score for idx, score in ann_recs}
    
    # Get co-occurrence recommendations
    co_occur_recs = get_co_occurrence_recommendations(playlist_idx, top_n*2)
    co_occur_dict = {idx: score for idx, score in co_occur_recs}
    
    # Combine scores
    combined_scores = {}
    all_tracks = set(list(ann_recs_dict.keys()) + list(co_occur_dict.keys()))
    
    for t in all_tracks:
        if t in current_tracks:
            continue
        ann_score = ann_recs_dict.get(t, 0)
        co_score = co_occur_dict.get(t, 0)
        combined_scores[t] = alpha * ann_score + (1-alpha) * co_score
    
    # Return top recommendations
    top_tracks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(idx_to_track[idx], score) for idx, score in top_tracks]

def process_challenge_playlists(output_dir='challenge_submissions'):
    """Process all challenge playlists and generate recommendations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each challenge playlist with num_samples > 0
    valid_playlists = challenge_playlists_df[challenge_playlists_df['num_samples'] > 0]
    results = []
    
    print(f"Processing {len(valid_playlists)} challenge playlists...")
    
    for index, row in valid_playlists.iterrows():
        playlist_id = row['playlist_id']
        name = row['name']
        num_tracks = row['num_tracks']
        num_samples = row['num_samples']
        
        print(f"Processing playlist: {name} (ID: {playlist_id}, Tracks: {num_tracks}, Samples: {num_samples})")
        
        try:
            playlist_idx = playlist_to_idx[playlist_id]
            recommendations = hybrid_recommendations(playlist_idx, alpha=0.7, top_n=500)
            
            # Save recommendations for this playlist
            for track_id, score in recommendations:
                results.append({
                    'playlist_id': playlist_id,
                    'track_id': track_id,
                    'score': score
                })
                
            print(f"Generated {len(recommendations)} recommendations for playlist {playlist_id}")
            
        except Exception as e:
            print(f"Error processing playlist {playlist_id}: {e}")
    
    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'challenge_recommendations.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Saved recommendations to {output_file}")
    
    # Create submission file with only playlist_id and track_id
    submission_df = results_df[['playlist_id', 'track_id']]
    submission_file = os.path.join(output_dir, 'challenge_submission.csv')
    submission_df.to_csv(submission_file, index=False)
    print(f"Saved submission file to {submission_file}")
    
    return results_df

# Main execution
if __name__ == "__main__":
    process_challenge_playlists()