"""Build matched real/fake pairs per generator: each real clip is cut to its fake's fps and duration.

Usage: python scripts/data/prepare_matched_models.py data/genvidbench/subset_profile.csv data/genvidbench/matched [per_model]
Every kept clip has a face in >= 80% of its frames and goes through the same ffmpeg settings
(480p, H.264, no audio). Writes <out>/<model>/{fake,real}/pairNN.mp4 and <out>/manifest.csv.
"""

import csv
import random
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

from rppg_detector.face import FaceDetector

MODELS = {"keling": "Kling", "ms": "ModelScope", "pika": "Pika", "svd": "SVD", "OpenAI_Sora": "Sora"}
MIN_FACE_RATE = 0.8
profile, out = pd.read_csv(sys.argv[1]), Path(sys.argv[2])
per_model = int(sys.argv[3]) if len(sys.argv) > 3 else 10
random.seed(0)
detector = FaceDetector()


def probe(path):
    """Return (fps, seconds) of a video file using ffprobe."""
    text = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=avg_frame_rate:format=duration", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.split()
    stream, duration = text[0].split(","), float(text[-1].split(",")[-1])
    num, den = (float(x) for x in stream[0].split("/"))
    return num / den, duration


def encode(src, dest, fps, seconds):
    """Cut src to `seconds` at `fps` with the shared settings."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", f"{seconds:.3f}", "-an",
                    "-vf", f"fps={fps:.4f},scale=-2:480", "-c:v", "libx264", "-crf", "23",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dest)], check=True)


def face_rate(path):
    """Fraction of frames in which MediaPipe finds a face."""
    cap, hits, total = cv2.VideoCapture(str(path)), 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        hits += detector.detect_faces(frame) is not None
    return hits / total if total else 0.0


real_pool = profile[(profile.source == "vript") & (profile.face_hits_of_5 >= 3)]
real_paths = list(real_pool.itertuples())
random.shuffle(real_paths)
rows = []
for source, model in MODELS.items():
    threshold = 2 if source == "OpenAI_Sora" else 3
    candidates = list(profile[(profile.source == source) & (profile.face_hits_of_5 >= threshold)].itertuples())
    random.shuffle(candidates)
    kept = 0
    for clip in candidates:
        if kept == per_model:
            break
        fake = out / model / "fake" / f"pair{kept:02d}.mp4"
        fps, duration = probe(clip.path)
        encode(clip.path, fake, fps, min(5.0, duration))
        fake_fps, fake_seconds = probe(fake)
        rate = face_rate(fake)
        if rate < MIN_FACE_RATE:
            fake.unlink()
            continue
        real = out / model / "real" / f"pair{kept:02d}.mp4"
        for candidate in real_paths:
            if candidate.seconds < fake_seconds + 0.2:
                continue
            encode(candidate.path, real, fake_fps, fake_seconds)
            real_rate = face_rate(real)
            real_paths.remove(candidate)
            if real_rate >= MIN_FACE_RATE:
                break
            real.unlink()
        else:
            fake.unlink()
            continue
        pair = f"{model}_{kept:02d}"
        rows.append((str(fake), 1, pair, model, fake_fps, fake_seconds, rate))
        rows.append((str(real), 0, pair, model, fake_fps, fake_seconds, real_rate))
        kept += 1
    print(f"{model}: kept {kept} pairs from {len(candidates)} candidates", flush=True)
with open(out / "manifest.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "label", "person_id", "model", "fps", "seconds", "face_rate"])
    writer.writerows(rows)
