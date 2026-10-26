"""Run the detector on matched real/fake pairs, one group per generator, and store everything.

Usage: python scripts/experiments/run_matched_models.py data/genvidbench/matched/manifest.csv
Each clip is analysed as ONE window covering the whole clip (clips are only 1.4-5 s long).
Writes results/matched_models_<timestamp>/{clips.csv,per_model.csv,pooled.json,summary.md,signals/}.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from rppg_detector.classify import DeepfakeClassifier
from rppg_detector.coherence import extract_window_features
from rppg_detector.face import FaceDetector
from rppg_detector.heartrate import estimate_heart_rate
from rppg_detector.signals import extract_rgb_signals, get_green_channel
from rppg_detector.video import read_video_frames

DETECTOR_FEATURES = ["mean_coherence", "min_coherence", "hr_spread_bpm", "quality"]
EXTRA_FEATURES = ["band_fraction", "frame_to_frame_std", "brightness"]
FEATURES = DETECTOR_FEATURES + EXTRA_FEATURES


def analyse(row, detector, signals_dir):
    """Return one dict of features for a whole clip (single window)."""
    detector.reset()
    frames, fps = read_video_frames(row.path)
    rgb = extract_rgb_signals(frames, detector)
    green = get_green_channel(rgb)
    np.savez_compressed(signals_dir / f"{Path(row.path).parent.parent.name}_{'fake' if row.label else 'real'}_{Path(row.path).stem}.npz",
                        rgb=rgb, fps=fps)
    seconds = len(frames) / fps
    window = extract_window_features(green, fps, seconds, seconds)[0]
    sos = butter(4, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    padlen = min(15, len(green) - 1)
    band = sosfiltfilt(sos, detrend(green, axis=0), axis=0, padlen=padlen)
    hr, _ = estimate_heart_rate(green, fps)
    return dict(path=row.path, model=row.model, label=row.label, pair=row.person_id, fps=fps,
                frames=len(frames), seconds=seconds, hr_bpm=hr,
                **dict(zip(DETECTOR_FEATURES, window)),
                band_fraction=float((band ** 2).mean() / max(green.var(), 1e-9)),
                frame_to_frame_std=float(np.diff(green, axis=0).std()), brightness=float(green.mean()))


def paired_summary(clips, features):
    """Per feature: AUC (fake=1), share of pairs with real > fake, sign-test and Wilcoxon p."""
    wide = clips.pivot(index="pair", columns="label", values=features)
    out = {}
    for feature in features:
        real, fake = wide[feature][0], wide[feature][1]
        diff = (real - fake).dropna()
        n_pos = int((diff > 0).sum())
        n = int((diff != 0).sum())
        out[feature] = {
            "pairs": int(len(diff)),
            "auc_fake_vs_real": float(roc_auc_score(clips.label, clips[feature])) if len(clips) > 2 else None,
            "pairs_real_higher": f"{n_pos}/{len(diff)}",
            "sign_test_p": float(binomtest(n_pos, n, 0.5).pvalue) if n else None,
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if n >= 5 else None,
            "mean_real": float(real.mean()), "mean_fake": float(fake.mean()),
        }
    return out


def cross_validate(clips, rounds_seed=None):
    """Out-of-fold fake probability, grouped by pair so a real/fake pair is never split."""
    X, y = clips[DETECTOR_FEATURES].values, clips.label.values
    scored = np.zeros(len(clips))
    splits = min(5, clips.pair.nunique())
    for train, test in GroupKFold(n_splits=splits).split(X, y, clips.pair.values):
        clf = DeepfakeClassifier()
        clf.train(X[train], y[train])
        scored[test] = clf.predict_proba(X[test])[:, 1]
    return scored


def main():
    manifest = pd.read_csv(sys.argv[1]).rename(columns={"person_id": "person_id"})
    out = Path("results") / f"matched_models_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / "signals").mkdir(parents=True)
    detector = FaceDetector()
    clips = pd.DataFrame([analyse(r, detector, out / "signals") for r in manifest.itertuples()])
    clips.to_csv(out / "clips.csv", index=False)

    per_model = {m: paired_summary(g, FEATURES) for m, g in clips.groupby("model") if g.pair.nunique() >= 3}
    pooled = paired_summary(clips, FEATURES)

    clips["fake_prob"] = cross_validate(clips)
    clips["pred"] = (clips.fake_prob > 0.5).astype(int)
    rng = np.random.default_rng(0)
    pairs = clips.pair.unique()
    boot = []
    for _ in range(1000):
        sample = rng.choice(pairs, len(pairs))
        picked = pd.concat([clips[clips.pair == p] for p in sample])
        boot.append(roc_auc_score(picked.label, picked.fake_prob))
    null = []
    for _ in range(200):
        shuffled = clips.copy()
        flip = dict(zip(pairs, rng.integers(0, 2, len(pairs))))  # swap labels within a pair
        shuffled["label"] = np.where(shuffled.pair.map(flip) == 1, 1 - shuffled.label, shuffled.label)
        null.append(roc_auc_score(shuffled.label, cross_validate(shuffled)))
    observed = roc_auc_score(clips.label, clips.fake_prob)
    classifier = {
        "clips": int(len(clips)), "pairs": int(len(pairs)),
        "accuracy": float((clips.pred == clips.label).mean()), "auc": float(observed),
        "auc_95ci_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "permutation_p": float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)),
        "accuracy_by_model": {m: float((g.pred == g.label).mean()) for m, g in clips.groupby("model")},
        "auc_by_model": {m: float(roc_auc_score(g.label, g.fake_prob)) for m, g in clips.groupby("model") if g.label.nunique() == 2 and len(g) >= 6},
    }
    clips.to_csv(out / "clips.csv", index=False)
    (out / "pooled.json").write_text(json.dumps({"classifier": classifier, "pooled_features": pooled,
                                                 "per_model_features": per_model}, indent=2))
    rows = [{"model": m, "feature": f, **v} for m, d in per_model.items() for f, v in d.items()]
    pd.DataFrame(rows).to_csv(out / "per_model.csv", index=False)
    lines = [f"# Matched real/fake pairs per generator\n",
             f"- {classifier['pairs']} pairs ({classifier['clips']} clips); each real clip cut to its fake's fps and duration; whole clip = one window",
             f"- pooled classifier (GroupKFold by pair): accuracy {classifier['accuracy']:.3f}, AUC {classifier['auc']:.3f} "
             f"(95% bootstrap CI {classifier['auc_95ci_bootstrap'][0]:.2f}-{classifier['auc_95ci_bootstrap'][1]:.2f}), permutation p = {classifier['permutation_p']:.3f}",
             f"- accuracy by model: {json.dumps(classifier['accuracy_by_model'])}",
             f"- AUC by model: {json.dumps(classifier['auc_by_model'])}", "", "## Pooled per-feature paired comparison", ""]
    lines += [f"- {f}: AUC {v['auc_fake_vs_real']:.3f}, real higher in {v['pairs_real_higher']} pairs, sign-test p {v['sign_test_p']:.3f}, "
              f"mean real {v['mean_real']:.3f} / fake {v['mean_fake']:.3f}" for f, v in pooled.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Results stored in {out}")
    print(json.dumps(classifier, indent=2))
    print("pooled features:")
    for f, v in pooled.items():
        print(f"  {f:20s} AUC {v['auc_fake_vs_real']:.3f}  real>fake in {v['pairs_real_higher']:>6}  sign p {v['sign_test_p']:.3f}  "
              f"real {v['mean_real']:.3f} fake {v['mean_fake']:.3f}")


if __name__ == "__main__":
    main()
