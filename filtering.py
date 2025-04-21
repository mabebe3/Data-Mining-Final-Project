import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import faiss
import time
import os
import glob
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras import ops

load_dotenv()

# Configure tensorflow for efficient resource usage

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# # Load training data
format_path = os.getenv("FORMATTED")
playlist_tracks_df = pd.read_csv(os.path.join(format_path,"playlists_tracks.csv"))
playlists_df = pd.read_csv(os.path.join(format_path,"playlists.csv"))
tracks_df = pd.read_csv(os.path.join(format_path,"tracks.csv"))
artists_df = pd.read_csv(os.path.join(format_path,"artists.csv"))

# # Load challenge data
challenge_path = os.getenv("ONLINE")
challenge_playlist_tracks_df = pd.read_csv(os.path.join(challenge_path,"playlists_tracks.csv"))
challenge_playlists_df = pd.read_csv(os.path.join(challenge_path,"playlists.csv"))

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

def train_autoencoder():
    """Train an autoencoder on SVD latent features and return compatible embeddings"""
    model_path = "latent_autoencoder.keras"
    
    # Use SVD latent features as our foundation
    latent_dim = track_factors.shape[1]  # k=100 from SVD
    num_tracks = len(unique_tracks)

    if os.path.exists(model_path):
        autoencoder = tf.keras.models.load_model(model_path, compile=False)
        print("Loaded pre-trained latent autoencoder.")
    else:
        print("Training autoencoder on SVD latent features...")
        tf.keras.backend.clear_session()

        # Build autoencoder using SVD features as input
        inputs = Input(shape=(latent_dim,))
        encoded = Dense(64, activation='relu')(inputs)
        decoded = Dense(latent_dim, activation='linear')(encoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train using SVD track factors as both input and target
        autoencoder.fit(
            x=track_factors,
            y=track_factors,
            epochs=15,
            batch_size=256
        )
        autoencoder.save(model_path)
        print("Autoencoder trained and saved.")

    # Create enhanced embeddings for all tracks
    encoder_input = autoencoder.input
    encoder_output = autoencoder.layers[1].output
    encoder = Model(encoder_input, encoder_output)

    track_embeddings = encoder.predict(track_factors, batch_size=4096)

    # Normalize embeddings for cosine similarity
    normalized = track_embeddings / np.linalg.norm(track_embeddings, axis=1, keepdims=True)
    normalized = normalized.astype('float32')
    dimension = normalized.shape[1]
    
    # Build FAISS index for autoencoder embeddings
    nlist = min(100, len(normalized) // 39)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(normalized)
    index.add(normalized)
    index.nprobe = 20

    return index, track_embeddings


def get_autoencoder_recommendations(playlist_idx, track_embeddings, ae_index, top_n=500):
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    if not current_tracks:
        return []
    
    # Get playlist embedding
    track_indices = list(current_tracks)
    avg_embedding = np.mean(track_embeddings[track_indices], axis=0)
    avg_embedding = avg_embedding.reshape(1, -1).astype('float32')
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding /= norm
    
    # Search FAISS index
    distances, indices = ae_index.search(avg_embedding, top_n + len(current_tracks))
    
    # Filter results
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(idx_to_track) and idx not in current_tracks:
            recommendations.append((idx, float(dist)))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]


def hybrid_recommendations(playlist_idx, model_weights, top_n, ae_index, track_embeddings):
    """Combine ANN and co-occurrence recommendations with weighting factor alpha"""
    current_tracks = set(playlist_to_tracks.get(playlist_idx, []))
    
    # Get ANN recommendations
    ann_recs = get_ann_recommendations(playlist_idx, top_n*2)
    ann_recs_dict = {idx: score for idx, score in ann_recs}
    
    # Get co-occurrence recommendations
    co_occur_recs = get_co_occurrence_recommendations(playlist_idx, top_n*2)
    co_occur_dict = {idx: score for idx, score in co_occur_recs}

    # Get auto-encoder recommendations
    ae_recs = get_autoencoder_recommendations(playlist_idx, track_embeddings, ae_index, top_n*2)
    ae_recs_dict = {idx: score for idx, score in ae_recs}
    
    # Combine scores
    combined_scores = {}
    all_tracks = set(list(ann_recs_dict.keys()) + list(co_occur_dict.keys())+ list(ae_recs_dict.keys()))
    
    for t in all_tracks:
        if t in current_tracks:
            continue
        ann_score = ann_recs_dict.get(t, 0)
        co_score = co_occur_dict.get(t, 0)
        ae_score = ae_recs_dict.get(t, 0)
        combined_scores[t] = model_weights[0] * ann_score + model_weights[1] * co_score + model_weights[2] * ae_score
    
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
    
    total = len(valid_playlists)
    print(f"Processing {total} challenge playlists...")
    progress = 0

    # Get ae model and embeddings
    ae_index, track_embeddings = train_autoencoder()
    
    for index, row in valid_playlists.iterrows():
        playlist_id = row['playlist_id']
        name = row['name']
        num_tracks = row['num_tracks']
        num_samples = row['num_samples']
        
        progress = progress + 1
        print(f"Processing playlist: {name}, {progress}/{total} (ID: {playlist_id}, Tracks: {num_tracks}, Samples: {num_samples})")
        
        try:
            playlist_idx = playlist_to_idx[playlist_id]
            recommendations = hybrid_recommendations(playlist_idx, [0.4, 0.2, 0.4], 500, ae_index, track_embeddings)
            
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
"""
python filtering.py
cd documents/github/data-mining-final-project
"""
if __name__ == "__main__":
    process_challenge_playlists() 