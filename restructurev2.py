import pandas as pd
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

# Load the challenge submission file
format_path = os.getenv("SUBMISSION")
df = pd.read_csv(os.path.join(format_path, "challenge_submission.csv"))

# Load track ID → URI mapping
track_uri_map = pd.read_csv("data/data_formatted/tracks.csv")[["track_id", "track_uri"]]
id_to_uri = dict(zip(track_uri_map.track_id, track_uri_map.track_uri))

# Group by playlist ID (pid), collecting track URIs in order
playlists = defaultdict(list)
for _, row in df.iterrows():
    pid = row['playlist_id']
    track_id = row['track_id']
    track_uri = id_to_uri.get(track_id, "unknown_uri")
    playlists[pid].append(track_uri)

# Create the final output file
with open('formatted_submission.csv', 'w', encoding='utf-8') as f:
    # Write the team info line
    f.write("team_info, my awesome team name, my_awesome_team@email.com\n\n")
    
    for pid in sorted(playlists.keys()):
        print(f"Working on{pid}")
        track_uris = playlists[pid]
        
        if len(track_uris) != 500:
            print(f"Warning: Playlist {pid} has {len(track_uris)} tracks (expected 500)")
        
        # Write the line in required format
        line = f"{pid}," + ",".join(track_uris)
        f.write(line + "\n")
