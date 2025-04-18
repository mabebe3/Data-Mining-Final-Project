def write_playlist(csv, pid, track_ids)
	if len(track_ids) != 500:
        raise ValueError("track_ids must have exactly 500 elements.")
    # Convert all track IDs to string (if not already)
    track_ids_str = [str(tid) for tid in track_ids]
    # Prepare the line
    line = f"{pid}, " + ", ".join(track_ids_str) + "\n"
    # Append to file
    with open(csv, 'a', encoding='utf-8') as f:
        f.write(line)

def write_team_info(csv, team_name, contact_email):
    with open(csv, 'w', encoding='utf-8') as f:
        f.write(f"team_info, {team_name}, {contact_email}\n")

# Input format: List of all challenge set playlists as a dictionary of pid and
# track_ids a length 500 list of track ids
csv_path = "submission.csv"
write_team_info(csv_path, "Kai Shu Nation", "zach.p.hammond@gmail.com")
for playlist in playlists:
	write_playlist(csv_path, playlist["pid"], playlist["track_ids"])