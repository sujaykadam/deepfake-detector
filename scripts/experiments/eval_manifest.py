"""GroupKFold check on a manifest: window-level and per-video accuracy, grouped by person_id."""

import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from rppg_detector.classify import DeepfakeClassifier
from rppg_detector.dataset import build_window_dataset, load_manifest_csv
from rppg_detector.face import FaceDetector

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--window", type=float, default=2.0)
parser.add_argument("--step", type=float, default=1.0)
parser.add_argument("--splits", type=int, default=5)
args = parser.parse_args()

manifest = load_manifest_csv(args.manifest)
X, y, groups = build_window_dataset(manifest, FaceDetector(), args.window, args.step)
print(f"{len(X)} windows from {len(manifest)} videos; real {int((y == 0).sum())}, fake {int((y == 1).sum())}")

window_acc = []
for train_idx, test_idx in GroupKFold(n_splits=args.splits).split(X, y, groups):
    assert not set(groups[train_idx]) & set(groups[test_idx])
    clf = DeepfakeClassifier(args.window, args.step)
    clf.train(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    window_acc.append(np.mean(pred == y[test_idx]))
print(f"window accuracy per fold: {np.round(window_acc, 3).tolist()}  mean {np.mean(window_acc):.3f}")
print(f"chance (majority class) = {max(np.mean(y), 1 - np.mean(y)):.3f}")
print("feature means  real:", X[y == 0].mean(axis=0).round(3).tolist())
print("feature means  fake:", X[y == 1].mean(axis=0).round(3).tolist())
