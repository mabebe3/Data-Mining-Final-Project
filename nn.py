import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import argparse

# ==============================================
# Build 1D CNN Architecture (for 2D input matrix)
# ==============================================
class PlaylistEmbeddingModel(Model):
    def __init__(self, input_features):
        super().__init__()
        self.cnn = tf.keras.Sequential([
            layers.Reshape((input_features, 1)),  # Explicitly add channel dimension
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim)  # Match song embedding dimension
        ])
    
    def call(self, x):
        return self.cnn(x)

# Custom loss
def cosine_similarity_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

# ==============================================
# Recommendation Generation
# ==============================================
def generate_recommendations(playlist_matrix, k=500):
    playlist_emb = tf.math.l2_normalize(model.predict(playlist_matrix), axis=1)
    song_emb = tf.math.l2_normalize(song_matrix, axis=1)
    similarity = tf.matmul(playlist_emb, song_emb, transpose_b=True)
    return tf.math.top_k(similarity, k=k).indices.numpy()

# python nn.py mpd.slice.0-999.json 1
if __name__ == "__main__":
    # Parse arguments (unchanged)
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str)
    parser.add_argument('playlist_index', type=int)
    args = parser.parse_args()

    # Load JSON and validate index (unchanged)
    with open(args.json_file, 'r') as f:
        playlists = json.load(f)
    if args.playlist_index < 0 or args.playlist_index >= len(playlists):
        raise ValueError(f"Invalid playlist index: {args.playlist_index}")

    # ==============================================
    # Load & Prepare Data (FIXED for 2D input)
    # ==============================================
    rname_matrix = np.load('svd_word_matrix_Vname.npy').astype(np.float32)  # Shape: (num_playlists, features)
    song_matrix = np.load('svd_song_name_vectors.npy').astype(np.float32)   # Shape: (num_songs, latent_dim)

    num_playlists, input_features = rname_matrix.shape  # Get 2D dimensions
    print(f"num_playlists: {num_playlists} {input_features}")
    num_songs, latent_dim = song_matrix.shape
    print(f"num_songs: {song_matrix} {latent_dim}")

    # ==============================================
    # Training Setup
    # ==============================================
    model = PlaylistEmbeddingModel(input_features=input_features)
    optimizer = tf.keras.optimizers.Adam(0.001)

    # Reshape input_data to (num_playlists, input_features, 1) for Conv1D
    dataset = tf.data.Dataset.from_tensor_slices(
        (rname_matrix, song_matrix[:num_playlists])  # Directly use 2D input
    ).shuffle(1024).batch(32).prefetch(2)

    model.compile(optimizer=optimizer, loss=cosine_similarity_loss)
    history = model.fit(dataset, epochs=20, callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ])

    # Generate recommendations (add batch dimension)
    selected_playlist = rname_matrix[args.playlist_index][np.newaxis, :]  # Shape: (1, features)
    recommendations = generate_recommendations(selected_playlist)
    print(f"Top 500 songs for playlist {args.playlist_index}: {recommendations[0]}")