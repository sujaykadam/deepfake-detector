"""Pair EVERY usable Kling clip with real Vript clips matched on fps, duration, encoding and face size.

Usage: python scripts/data/prepare_kling_full.py data/genvidbench/subset data/genvidbench/subset_profile.csv data/genvidbench/kling_full [reals_per_fake]
Face size = median height of the face landmarks as a fraction of the frame height. A clip is usable
if a face is found in >= 80% of the sampled frames. Real clips are matched to within +-30% face size,
and a real YouTube video is used for only one fake (so one group never straddles a split).
Writes <out>/{fake,real}/*.mp4 and <out>/manifest.csv (path,label,person_id,subtype,fps,seconds,face_rate,face_size).
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rppg_detector.face import FaceDetector

subset, profile_csv, out = Path(sys.argv[1]), pd.read_csv(sys.argv[2]), Path(sys.argv[3])
reals_per_fake = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SECONDS, MIN_RATE, TOLERANCE = 5.0, 0.8, 0.3
detector = FaceDetector()


def probe_fps(path):
    """Return the frame rate (frames per second) of a video."""
    text = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()
    num, den = (float(x) for x in text.split("/"))
    return num / den


def face_stats(path, seconds=SECONDS, step=5):
    """Return (fraction of sampled frames with a face, median face height / frame height)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hits, heights, sampled = 0, [], 0
    for index in range(0, int(seconds * fps), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        sampled += 1
        landmarks = detector.detect_faces(frame)
        if landmarks is not None:
            hits += 1
            heights.append(np.ptp(landmarks[:, 1]) / frame.shape[0])
    return (hits / sampled if sampled else 0.0), (float(np.median(heights)) if heights else 0.0)


def encode(src, dest, fps):
    """Cut src to SECONDS at `fps` with the shared encoding settings."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(SECONDS), "-an",
                    "-vf", f"fps={fps:.4f},scale=-2:480", "-c:v", "libx264", "-crf", "23",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dest)], check=True)


fakes = []
for path in sorted((subset / "keling").rglob("*")):
    if path.suffix.lower() in (".mp4", ".mov") and "/._" not in str(path):
        rate, size = face_stats(path)
        if rate >= MIN_RATE:
            fakes.append((path, rate, size))
print(f"Kling: {len(fakes)} usable clips", flush=True)

pool = profile_csv[(profile_csv.source == "vript") & (profile_csv.face_hits_of_5 >= 2) & (profile_csv.seconds >= SECONDS + 0.2)]
real_stats = []
for clip in pool.itertuples():
    rate, size = face_stats(clip.path)
    if rate >= MIN_RATE:
        real_stats.append((Path(clip.path), size, re.sub(r"-Scene-\d+$", "", Path(clip.path).stem)))
print(f"Vript: {len(real_stats)} usable real clips of {len(pool)}", flush=True)

rows, used_ids, used_paths = [], set(), set()
for number, (src, rate, size) in enumerate(sorted(fakes, key=lambda f: f[2])):
    subtype = src.parent.name
    pair = f"kling_{number:03d}"
    options = sorted((r for r in real_stats if r[0] not in used_paths and r[2] not in used_ids
                      and abs(r[1] / size - 1) <= TOLERANCE), key=lambda r: abs(r[1] / size - 1))
    chosen, ids = [], set()
    for real in options:
        if real[2] not in ids:
            chosen.append(real)
            ids.add(real[2])
        if len(chosen) == reals_per_fake:
            break
    if not chosen:
        continue
    fps = probe_fps(src)
    fake_dest = out / "fake" / f"{pair}_{subtype}.mp4"
    encode(src, fake_dest, fps)
    rows.append((str(fake_dest), 1, pair, subtype, fps, SECONDS, rate, size))
    for k, (real_src, real_size, youtube_id) in enumerate(chosen):
        real_dest = out / "real" / f"{pair}_real{k}.mp4"
        encode(real_src, real_dest, fps)
        rows.append((str(real_dest), 0, pair, "real", fps, SECONDS, 1.0, real_size))
        used_ids.add(youtube_id)
        used_paths.add(real_src)
with open(out / "manifest.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "label", "person_id", "subtype", "fps", "seconds", "face_rate", "face_size"])
    writer.writerows(rows)
frame = pd.DataFrame(rows, columns=["path", "label", "person_id", "subtype", "fps", "seconds", "face_rate", "face_size"])
print(frame.groupby("label").agg(clips=("path", "size"), size=("face_size", "median")).to_string())
