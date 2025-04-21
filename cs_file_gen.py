import json
import os
import csv
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Paths
MPD_PATH = "C:\\Education\\Senior Spring\\CS 470\\Data-Mining-Final-Project\\data\\raw\\data\\"
OUTPUT_SONGS_CSV = "song_titles.csv"
OUTPUT_PLAYLISTS_CSV = "playlist_titles.csv"

# Data structures
song_to_titles = defaultdict(list)
playlist_ids = []
playlist_titles = []

# Process all slice files
slice_files = sorted(glob(os.path.join(MPD_PATH, "mpd.slice.*.json")))
print(f"Processing {len(slice_files)} slice files...")

for slice_file in tqdm(slice_files):
    with open(slice_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for playlist in data["playlists"]:
            pid = playlist["pid"]
            title = playlist["name"].strip().lower()

            playlist_ids.append(pid)
            playlist_titles.append(title)

            for track in playlist["tracks"]:
                song_to_titles[track["track_uri"]].append(title)

# Write song-track grouped CSV
with open(OUTPUT_SONGS_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["track_uri", "playlist_titles"])
    for track_uri, titles in song_to_titles.items():
        writer.writerow([track_uri, list(set(titles))])

# Write playlist-title CSV
with open(OUTPUT_PLAYLISTS_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["playlist_id", "playlist_title"])
    writer.writerows(zip(playlist_ids, playlist_titles))

print("CSV writing complete. Creating matrix...")
