"""Profile clips (fps, duration, size) and test for a face on 5 sampled frames.

Usage: python scripts/data/profile_clips.py data/genvidbench/subset out.csv
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from rppg_detector.face import FaceDetector

root, out = Path(sys.argv[1]), sys.argv[2]
detector = FaceDetector()
clips = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".mov"))
with open(out, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "source", "fps", "frames", "seconds", "width", "height", "face_hits_of_5"])
    for i, path in enumerate(clips, 1):
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hits = 0
        if n > 0:
            for index in np.linspace(0, n - 1, 5).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                hits += bool(ok and detector.detect_faces(frame) is not None)
        writer.writerow([path, path.relative_to(root).parts[0], fps, n, n / fps if fps else 0,
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), hits])
        cap.release()
        if i % 200 == 0:
            print(f"{i}/{len(clips)}", flush=True)
