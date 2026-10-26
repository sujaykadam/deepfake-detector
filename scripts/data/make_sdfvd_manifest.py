"""Write manifest.csv for the extracted SDFVD folder: vN (real) pairs with vsN (fake)."""

import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])  # e.g. data/sdfvd/SDFVD
out = Path(sys.argv[2])
rows = []
for label, folder, pattern in ((0, "videos_real", r"v(\d+)\.mp4"), (1, "videos_fake", r"vs(\d+)\.mp4")):
    for path in sorted((root / folder).glob("*.mp4")):
        match = re.fullmatch(pattern, path.name)
        if match:
            rows.append((str(path), label, int(match.group(1))))
with open(out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "person_id"])
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
