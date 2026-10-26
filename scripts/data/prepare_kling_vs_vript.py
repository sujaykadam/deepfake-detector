"""Cut qualifying Kling (fake) and Vript (real) clips to 5 s with identical ffmpeg settings.

Usage: python scripts/data/prepare_kling_vs_vript.py data/genvidbench/subset_profile.csv data/genvidbench/kling_vs_vript
Writes <out>/{real,fake}/*.mp4 and <out>/manifest.csv (path,label,person_id,source,subtype).
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

profile, out = pd.read_csv(sys.argv[1]), Path(sys.argv[2])
profile = profile[(profile.face_hits_of_5 >= 4) & (profile.seconds >= 5.0)]
rows = []
for source, label, folder in (("keling", 1, "fake"), ("vript", 0, "real")):
    (out / folder).mkdir(parents=True, exist_ok=True)
    for src in profile[profile.source == source].path:
        stem = Path(src).stem
        if source == "vript":  # clips of one YouTube video share a person_id
            person, subtype = "yt_" + re.sub(r"-Scene-\d+$", "", stem), "real"
        else:
            subtype = Path(src).parent.name  # T2V or I2V
            person = f"kling_{subtype}_{stem}"
        dest = out / folder / f"{re.sub(r'[^A-Za-z0-9_-]', '_', person)}_{stem[-12:]}.mp4".replace("__", "_")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", "5", "-an",
                        "-vf", "fps=25,scale=-2:480", "-c:v", "libx264", "-crf", "23",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dest)], check=True)
        rows.append((str(dest), label, person, source, subtype))
with open(out / "manifest.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "label", "person_id", "source", "subtype"])
    writer.writerows(rows)
print(f"wrote {len(rows)} clips; manifest at {out / 'manifest.csv'}")
