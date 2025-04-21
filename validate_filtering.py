import pandas as pd
import sys

def validate_submission(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to load submission file: {e}")
        sys.exit(1)

    if not {'playlist_id', 'track_id'}.issubset(df.columns):
        print("❌ Submission file must contain 'playlist_id' and 'track_id' columns.")
        sys.exit(1)

    # Count the number of recommendations per playlist
    playlist_counts = df['playlist_id'].value_counts()
    invalid = playlist_counts[playlist_counts != 500]

    if not invalid.empty:
        print("❌ Some playlists do not have exactly 500 recommendations:")
        print(invalid)
        sys.exit(1)
    else:
        print("✅ Submission is valid. All playlists have exactly 500 recommendations.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <path_to_submission_file>")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    validate_submission(submission_file)
