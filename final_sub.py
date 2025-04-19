import gzip

# Read raw submission (assumes it's already formatted correctly)
with open("submission_raw.csv", "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

# Read submission.csv, skip first row and first column in each row
with open("submission.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]  # skip team_info line

# Process lines: remove first column in each (label "pid")
processed_lines = []
for line in lines:
    parts = line.strip().split(',')
    if len(parts) > 1:
        new_line = ','.join(parts[1:])  # drop 'pid'
        processed_lines.append(new_line + '\n')

# Concatenate
final_lines = raw_lines + processed_lines

# Write to CSV
with open("final_submission.csv", "w", encoding="utf-8") as f:
    f.writelines(final_lines)

# Compress to .csv.gz
with open("final_submission.csv", "rt", encoding="utf-8") as f_in:
    with gzip.open("final_submission.csv.gz", "wt", encoding="utf-8") as f_out:
        f_out.writelines(f_in)

print("final_submission.csv and final_submission.csv.gz written.")
