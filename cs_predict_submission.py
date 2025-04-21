import numpy as np
import pandas as pd
import json
import csv
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os
from tqdm import tqdm

# === ENV SETUP === #
load_dotenv()
CHALLENGE = os.getenv("CHALLENGE")
json_path = os.path.join(CHALLENGE, "challenge_set.json")

# === CONFIGURATION === #
TOP_K = 500
NUM_PLAYLISTS = 1000000  # From training set
NUM_TO_PROCESS = 1000
RAW_SUBMISSION_CSV = "submission_raw.csv"
SUBMISSION_GZ_FILE = "submission.csv.gz"
TEAM_NAME = "Kai Shu Nation"
TEAM_EMAIL = "KaiShuNation@gmail.com"

# === LOAD CHALLENGE JSON === #
print(f"Loading playlists from {json_path}...")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    all_playlists = data["playlists"]
    challenge_playlists = all_playlists[:NUM_TO_PROCESS]
print(f"Using first {len(challenge_playlists)} playlists for submission.")

# === LOAD TRAINED MODELS AND DATA === #
print("Loading latent matrices and training playlist titles...")
U = np.load("Rname_U.npy")
Vt = np.load("Rname_Vt.npy")
song_df = pd.read_csv("song_titles.csv")
song_uris = song_df["track_uri"].tolist()

playlist_df = pd.read_csv("playlist_titles.csv")
training_titles = playlist_df["playlist_title"].astype(str).tolist()
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(training_titles)

# === RECOMMENDER FUNCTION === #
def recommend_from_name(playlist_name, top_k=500):
    name_vector = vectorizer.transform([playlist_name])
    embedding = name_vector @ Vt.T  # (1, k)
    song_embeddings = U[NUM_PLAYLISTS:]
    scores = song_embeddings @ embedding.T
    scores = scores.flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [song_uris[i] for i in top_indices]

# === WRITE RAW CSV FIRST === #
print(f"Writing raw CSV to {RAW_SUBMISSION_CSV}...")
with open(RAW_SUBMISSION_CSV, "w", newline="", encoding="utf-8") as raw_csv:
    raw_writer = csv.writer(raw_csv)
    raw_writer.writerow(["team_info", TEAM_NAME, TEAM_EMAIL])
    raw_writer.writerow([])

    for entry in tqdm(challenge_playlists, desc="Generating recommendations"):
        pid = entry["pid"]
        name = entry.get("name", "").strip().lower()
        top_uris = recommend_from_name(name, TOP_K)
        raw_writer.writerow([pid] + top_uris)

# === COMPRESS CSV TO GZ === #
print(f"Compressing {RAW_SUBMISSION_CSV} to {SUBMISSION_GZ_FILE}...")
with open(RAW_SUBMISSION_CSV, "rt", encoding="utf-8") as raw_file:
    with gzip.open(SUBMISSION_GZ_FILE, "wt", encoding="utf-8") as gz_file:
        gz_file.writelines(raw_file)

print("✅ Submission complete!")
print(f"Saved raw: {RAW_SUBMISSION_CSV}")
print(f"Saved compressed: {SUBMISSION_GZ_FILE}")
