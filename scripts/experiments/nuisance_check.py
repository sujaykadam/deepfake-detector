"""Check whether simple non-pulse signal statistics separate real from fake as well as the detector.

Usage: python scripts/experiments/nuisance_check.py results/<run> manifest.csv
Reads the cached signals/*.npz of a run_experiment.py run; writes nuisance_features.csv and nuisance_auc.json there.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt
from sklearn.metrics import roc_auc_score

run, manifest = Path(sys.argv[1]), pd.read_csv(sys.argv[2]).set_index("path")
videos = pd.read_csv(run / "videos.csv")
rows = []
for video in videos[videos.face_found].itertuples():
    name = f"{'fake' if video.label else 'real'}_{Path(video.path).stem}"
    data = np.load(run / "signals" / f"{name}.npz")
    green, fps = data["rgb"][:, :, 1], float(data["fps"])
    sos = butter(4, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    band = sosfiltfilt(sos, detrend(green, axis=0), axis=0)
    rows.append({"path": video.path, "label": video.label, "brightness": green.mean(),
                 "raw_std": green.std(), "frame_to_frame_std": np.diff(green, axis=0).std(),
                 "band_power": (band ** 2).mean(), "band_fraction": (band ** 2).mean() / max(green.var(), 1e-9)})
table = pd.DataFrame(rows)
table.to_csv(run / "nuisance_features.csv", index=False)
aucs = {c: float(roc_auc_score(table.label, table[c])) for c in table.columns[2:]}
(run / "nuisance_auc.json").write_text(json.dumps(aucs, indent=2))
print(json.dumps(aucs, indent=2))
