import pandas as pd
import gzip
import os
from dotenv import load_dotenv
load_dotenv()
# Load submission file

format_path = os.getenv("SUBMISSION")
df = pd.read_csv(os.path.join(format_path,"challenge_submission.csv"))
# df = pd.read_csv("./challenge_submission.csv")

# Group by playlist and aggregate track URIs
grouped = df.groupby("playlist_id")["track_id"].apply(list)

# Ensure each playlist has exactly 500 unique track URIs
grouped = grouped.apply(lambda tracks: list(dict.fromkeys(tracks))[:500])

# Load track ID → URI mapping
track_uri_map = pd.read_csv("data/data_formatted/tracks.csv")[["track_id", "track_uri"]]
id_to_uri = dict(zip(track_uri_map.track_id, track_uri_map.track_uri))

# Format lines
lines = []
lines.append("team_info, Kai Shu Nation, KaiShuNation@gmail.com")

for pid, track_ids in grouped.items():
    track_uris = [id_to_uri.get(tid, "unknown_uri") for tid in track_ids]
    if len(track_uris) == 500:
        line = f"pid,{pid}," + ",".join(track_uris)
        lines.append(line)
    else:
        print(f"Warning: playlist {pid} has {len(track_uris)} URIs")

# Save to uncompressed CSV
with open("submission.csv", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

# Compress to .csv.gz
with open("submission.csv", "rt", encoding="utf-8") as f_in:
    with gzip.open("submission.csv.gz", "wt", encoding="utf-8") as f_out:
        f_out.writelines(f_in)

print("submission.csv and submission.csv.gz written.")
