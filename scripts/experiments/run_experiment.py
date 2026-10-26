"""Run the detector on every manifest video and store all intermediate data and metrics.

Usage: python scripts/experiments/run_experiment.py data/sdfvd/manifest.csv --window 2 --step 1
Writes results/<name>_<timestamp>/ with videos.csv, windows.csv, cv_predictions.csv,
metrics.json, summary.md, config.json, model.joblib and signals/<video>.npz.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from rppg_detector.classify import DeepfakeClassifier
from rppg_detector.coherence import extract_window_features
from rppg_detector.dataset import load_manifest_csv
from rppg_detector.face import FaceDetector
from rppg_detector.heartrate import estimate_heart_rate, estimate_heart_rate_per_patch
from rppg_detector.signals import extract_rgb_signals, get_green_channel
from rppg_detector.video import read_video_frames

FEATURES = ["mean_coherence", "min_coherence", "hr_spread_bpm", "quality"]


class CountingDetector(FaceDetector):
    """FaceDetector that counts frames where MediaPipe found no face."""

    misses = 0

    def detect_faces(self, frame):
        landmarks = super().detect_faces(frame)
        self.misses += landmarks is None
        return landmarks


def process_video(row, detector, args, signals_dir):
    """Return (video record, window feature rows) for one manifest row."""
    detector.reset()
    detector.misses = 0
    frames, fps = read_video_frames(row.path)
    rgb = extract_rgb_signals(frames, detector)
    record = {"path": row.path, "label": row.label, "person_id": row.person_id,
              "fps": fps, "frames": len(frames), "seconds": len(frames) / fps,
              "height": frames.shape[1], "width": frames.shape[2],
              "frames_without_face": detector.misses, "face_found": bool(rgb.any())}
    if not rgb.any():
        return record, []
    name = f"{'fake' if row.label else 'real'}_{Path(row.path).stem}"
    np.savez_compressed(signals_dir / f"{name}.npz", rgb=rgb, fps=fps)
    green = get_green_channel(rgb)
    hr, quality = estimate_heart_rate(green, fps)
    patch_hr = estimate_heart_rate_per_patch(green, fps)
    record.update(hr_bpm=hr, hr_quality=quality, patch_hr_spread_bpm=float(np.ptp(patch_hr)))
    windows = extract_window_features(green, fps, args.window, args.step)
    rows = [dict(zip(FEATURES, w), path=row.path, label=row.label, person_id=row.person_id,
                 window=i) for i, w in enumerate(windows)]
    return record, rows


def cross_validate(windows, args):
    """Out-of-fold window and video predictions with GroupKFold by person_id."""
    X, y, groups = windows[FEATURES].values, windows.label.values, windows.person_id.values
    windows = windows.copy()
    windows["fake_prob"] = np.nan
    splits = min(args.splits, len(np.unique(groups)))
    fold_acc = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=splits).split(X, y, groups)):
        assert not set(groups[train]) & set(groups[test]), "person_id on both sides"
        clf = DeepfakeClassifier(args.window, args.step)
        clf.train(X[train], y[train])
        windows.iloc[test, windows.columns.get_loc("fake_prob")] = clf.predict_proba(X[test])[:, 1]
        fold_acc.append(float(np.mean(clf.predict(X[test]) == y[test])))
    return windows, fold_acc


def permutation_pvalue(windows, args, observed_auc, rounds=200):
    """Share of label-shuffled runs (labels permuted across videos) that match the real AUC."""
    rng = np.random.default_rng(0)
    videos = windows.drop_duplicates("path")[["path", "label"]]
    hits = 0
    for _ in range(rounds):
        shuffled = dict(zip(videos.path, rng.permutation(videos.label.values)))
        trial = windows.copy()
        trial["label"] = trial.path.map(shuffled)
        scored, _ = cross_validate(trial, args)
        hits += roc_auc_score(trial.label, scored.fake_prob) >= observed_auc
    return (hits + 1) / (rounds + 1)


def bootstrap_auc_ci(per_video, rounds=1000):
    """95% bootstrap interval of the video-level AUC, resampling person_id groups."""
    rng = np.random.default_rng(0)
    groups = {g: d for g, d in per_video.groupby("person_id")}
    ids, aucs = list(groups), []
    for _ in range(rounds):
        sample = pd.concat([groups[g] for g in rng.choice(ids, len(ids))])
        if sample.label.nunique() == 2:
            aucs.append(roc_auc_score(sample.label, sample.mean_fake_prob))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def paired_tests(videos):
    """Compare each real clip with its fake: does a fake differ in heart rate or quality?"""
    found = videos[videos.face_found]
    real = found[found.label == 0].set_index("person_id")
    fake = found[found.label == 1].set_index("person_id")
    both = real.index.intersection(fake.index)
    out = {"pairs": int(len(both))}
    if len(both) < 2:
        return out
    for column in ("hr_bpm", "hr_quality"):
        diff = (real.loc[both, column] - fake.loc[both, column]).values
        pvalue = float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        out[column] = {"mean_real": float(real.loc[both, column].mean()),
                       "mean_fake": float(fake.loc[both, column].mean()),
                       "mean_abs_pair_diff": float(np.abs(diff).mean()),
                       "wilcoxon_p": pvalue}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--name", default="sdfvd")
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()

    out = Path(args.out_root) / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / "signals").mkdir(parents=True)
    manifest = load_manifest_csv(args.manifest)
    detector = CountingDetector()

    records, window_rows = [], []
    for i, row in enumerate(manifest.itertuples(), 1):
        record, rows = process_video(row, detector, args, out / "signals")
        records.append(record)
        window_rows += rows
        print(f"[{i}/{len(manifest)}] {Path(row.path).name}: "
              f"{'ok' if record['face_found'] else 'NO FACE'}", flush=True)
    videos, windows = pd.DataFrame(records), pd.DataFrame(window_rows)
    videos.to_csv(out / "videos.csv", index=False)

    windows, fold_acc = cross_validate(windows, args)
    windows.to_csv(out / "windows.csv", index=False)
    windows["pred"] = (windows.fake_prob > 0.5).astype(int)
    per_video = windows.groupby("path").agg(label=("label", "first"), person_id=("person_id", "first"),
                                            windows=("pred", "size"), fake_fraction=("pred", "mean"),
                                            mean_fake_prob=("fake_prob", "mean")).reset_index()
    per_video["pred"] = (per_video.fake_fraction > 0.5).astype(int)
    per_video.to_csv(out / "cv_predictions.csv", index=False)

    y = windows.label.values
    metrics = {
        "videos_total": int(len(videos)), "videos_with_face": int(videos.face_found.sum()),
        "videos_without_face": videos[~videos.face_found].path.tolist(),
        "windows": int(len(windows)), "windows_real": int((y == 0).sum()), "windows_fake": int((y == 1).sum()),
        "chance_window_accuracy": float(max(y.mean(), 1 - y.mean())),
        "fold_window_accuracy": fold_acc, "mean_window_accuracy": float(np.mean(fold_acc)),
        "window_roc_auc": float(roc_auc_score(y, windows.fake_prob)),
        "window_auc_permutation_p": permutation_pvalue(windows, args, roc_auc_score(y, windows.fake_prob)),
        "video_accuracy": float(np.mean(per_video.pred == per_video.label)),
        "video_balanced_accuracy": float(balanced_accuracy_score(per_video.label, per_video.pred)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y, windows.pred)),
        "video_roc_auc": float(roc_auc_score(per_video.label, per_video.mean_fake_prob)),
        "video_roc_auc_95ci": bootstrap_auc_ci(per_video),
        "feature_auc_fake_vs_real": {f: float(roc_auc_score(y, windows[f])) for f in FEATURES},
        "feature_means": {k: windows[windows.label == v][FEATURES].mean().round(4).to_dict()
                          for k, v in (("real", 0), ("fake", 1))},
        "paired": paired_tests(videos),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    final = DeepfakeClassifier(args.window, args.step)
    final.train(windows[FEATURES].values, y)
    final.save(str(out / "model.joblib"))

    config = {"manifest": args.manifest, "window_seconds": args.window, "step_seconds": args.step,
              "splits": args.splits, "features": FEATURES, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "face_model": detector._model_path, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    (out / "summary.md").write_text(
        f"# {args.name} experiment\n\n"
        f"- videos with a face: {metrics['videos_with_face']}/{metrics['videos_total']}\n"
        f"- windows: {metrics['windows']} (real {metrics['windows_real']}, fake {metrics['windows_fake']}), "
        f"window {args.window}s step {args.step}s\n"
        f"- GroupKFold({min(args.splits, windows.person_id.nunique())}) by person_id\n"
        f"- window accuracy {metrics['mean_window_accuracy']:.3f} (chance {metrics['chance_window_accuracy']:.3f}), "
        f"ROC AUC {metrics['window_roc_auc']:.3f} (permutation p = {metrics['window_auc_permutation_p']:.3f})\n"
        f"- video accuracy (majority vote) {metrics['video_accuracy']:.3f}, balanced {metrics['video_balanced_accuracy']:.3f}, ROC AUC {metrics['video_roc_auc']:.3f}\n"
        f"- video ROC AUC 95% CI {metrics['video_roc_auc_95ci'][0]:.3f}-{metrics['video_roc_auc_95ci'][1]:.3f}\n"
        f"- window balanced accuracy {metrics['window_balanced_accuracy']:.3f}\n"
        f"- per-feature ROC AUC (0.5 = no signal): {metrics['feature_auc_fake_vs_real']}\n"
        f"- paired real-vs-fake: {metrics['paired']}\n")
    print(f"\nResults stored in {out}")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
