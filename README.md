# rppg_detector

A learning-project **deepfake detector based on remote photoplethysmography (rPPG)**, the idea behind
Intel's FakeCatcher. It turns a face video into an estimated heart rate and a real-or-fake score,
using classical signal processing and a logistic regression. CPU only, CLI only, no deep learning.

> **Status: a working pipeline, not a working detector.** Everything runs end to end on real video
> (face finding, pulse extraction, training, prediction, grouped cross-validation, 55 passing tests).
> But the measured ability to tell real from AI-generated video is weak and mostly explained by
> something other than a pulse. See [What the experiments found](#what-the-experiments-found), and
> read [Known limits](#known-limits) before trusting any score.

---

## Contents

1. [The idea](#the-idea)
2. [How the pipeline works](#how-the-pipeline-works)
3. [Install](#install)
4. [Usage](#usage)
5. [Tests and checks](#tests-and-checks)
6. [Project layout](#project-layout)
7. [Datasets](#datasets)
8. [Reproducing the experiments](#reproducing-the-experiments)
9. [What the experiments found](#what-the-experiments-found)
10. [Known limits](#known-limits)
11. [Differences from the original brief](#differences-from-the-original-brief)
12. [Ideas for improving it](#ideas-for-improving-it)

---

## The idea

Every heartbeat pushes blood through the skin and changes, very slightly, how much green light the
skin reflects. A camera records that as a colour flicker of well under one percent. On a **real**
face, the flicker is the same heartbeat everywhere, so the signals from the forehead, left cheek and
right cheek **agree**. On a **fake** face, the pixels are synthesised or pasted in and the three
signals tend to agree less.

The detector measures that agreement:

1. find the face in every frame and cut out three skin patches,
2. take the average green value of each patch over time (three signals),
3. keep only frequencies a heart can produce (0.7 to 3.0 Hz, which is 42 to 180 beats per minute),
4. measure how closely the three signals agree, and how strong and consistent the pulse is,
5. feed those numbers to a logistic regression that says real or fake.

---

## How the pipeline works

```
video file
   |  video.py        cv2.VideoCapture: frames + the real frame rate (fps)
   v
frames  --->  face.py       MediaPipe Face Landmarker: 478 landmarks per frame,
   |                        three skin patches (forehead, left cheek, right cheek)
   v
signals.py     mean R,G,B of each patch per frame -> (frames, patches, 3); green channel by default
   |
   v
heartrate.py   detrend -> band-pass 0.7-3.0 Hz -> FFT peak -> beats per minute + quality
   |
   v
coherence.py   sliding windows (default 10 s, step 5 s) -> 4 features per window
   |
   v
classify.py    logistic regression on window features; every window votes, majority wins
```

| Module | What it does | Units |
|---|---|---|
| `video.py` | Reads frames with OpenCV and returns the **real** frame rate with them; raises a clear error for unreadable files. | fps (Hz) |
| `face.py` | Wraps MediaPipe Face Landmarker. Defines the three skin patches as landmark index lists in one constants block. Missing faces carry the previous landmarks forward and count as dropped frames. The model file is loaded lazily, so tests can stub the detector. | pixels |
| `signals.py` | Mean colour per patch per frame; green channel is the default method. `extend_signal_methods` is the extension point for CHROM and POS (not implemented). `video_to_green_signals` is the one-call video to signal helper. | 0 to 255 |
| `heartrate.py` | `detrend`, 4th-order Butterworth band-pass (`sosfiltfilt`), `numpy.fft.rfft` peak, and a quality score (peak power / band power). | Hz, BPM |
| `coherence.py` | `scipy.signal.coherence` between every pair of patches inside the heart-rate band, per sliding window. | s, Hz |
| `classify.py` | `DeepfakeClassifier`: scaling plus class-balanced logistic regression, majority vote over windows, `joblib` save and load (window settings are saved with the model). | |
| `dataset.py` | Reads the manifest CSV and turns each video into window features. | |
| `evaluate.py` | Heart-rate mean absolute error, and `GroupKFold` cross-validation grouped by person. Never a random split. | BPM |
| `cli.py` | `argparse` front end with three commands: `hr`, `train`, `predict`. | |

**The four features of each window**

| Feature | Meaning |
|---|---|
| mean pairwise coherence | average agreement of the three patch pairs (0 to 1) |
| minimum pairwise coherence | agreement of the worst pair |
| heart-rate spread | highest minus lowest per-patch heart-rate estimate (BPM) |
| quality | share of signal variance that the patches have in common (0 to 1) |

---

## Install

Requires **Python 3.11 or newer**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

With `uv`: `uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[test]"`.

Dependencies are bounded in `pyproject.toml` to the versions the project was tested with
(OpenCV 5.0, MediaPipe 1.1, NumPy 2.4, SciPy 1.17, scikit-learn 1.9, pandas 3.1).

### The face model (one-off download)

MediaPipe needs a model file. The package never downloads anything itself, so fetch it once:

```bash
mkdir -p data/models
curl -L -o data/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
export RPPG_FACE_MODEL=$PWD/data/models/face_landmarker.task
```

Either set `RPPG_FACE_MODEL`, or pass `--face-model PATH` to every command. Without a model the
commands stop with a message saying so.

---

## Usage

### Estimate a heart rate

```bash
rppg_detector hr VIDEO [--face-model PATH]
```

```
Loaded video: clip.mp4
FPS: 30.00
Frames: 900
Estimated heart rate: 66.7 BPM
Quality score: 0.2212
```

If no frame contains a face, the command exits with `Error: No face found in any frame of ...`.

### Train a classifier

```bash
rppg_detector train MANIFEST_CSV --out model.joblib [--window 10] [--step 5] [--face-model PATH]
```

The manifest has the columns `path,label,person_id`; `label` is **0 for real and 1 for fake**:

```csv
path,label,person_id
data/sdfvd/SDFVD/videos_real/v1.mp4,0,1
data/sdfvd/SDFVD/videos_fake/vs1.mp4,1,1
```

`person_id` is what the evaluation keeps together: every clip of one person, and every fake made
from one real video, must share an ID so they never land on both sides of a train/test split.

`--window` and `--step` are in **seconds** (defaults 10 and 5). Clips shorter than the window give
no usable windows, so use shorter windows for short clips; the values are stored inside the model so
`predict` uses the same ones. Videos with no detectable face are skipped with a message.

### Classify a video

```bash
rppg_detector predict VIDEO --model model.joblib [--face-model PATH]
```

```
Prediction: real
Score (fraction of windows voting fake): 0.2500
```

### From Python

```python
from rppg_detector.face import FaceDetector
from rppg_detector.signals import video_to_green_signals
from rppg_detector.heartrate import estimate_heart_rate

signals, fps = video_to_green_signals("clip.mp4", FaceDetector("data/models/face_landmarker.task"))
bpm, quality = estimate_heart_rate(signals, fps)
```

---

## Tests and checks

```bash
pytest                # 55 tests, no network, no video files, no MediaPipe model needed
./check.sh            # runs the project's success criteria end to end
```

The tests generate their own data: a 1.2 Hz sine (72 BPM) with drift and noise must come back as
72 BPM within 3; three patches sharing a 1.2 Hz pulse must score clearly higher coherence than
three unrelated ones; a synthetic `cv2.VideoWriter` clip must be read back with the right fps and
frame count (the face finder is stubbed); the classifier must reach at least 90% on separable
synthetic features; and `GroupKFold` must never put one `person_id` on both sides of a split.

`check.sh` also verifies the install, the 72 BPM accuracy, the coherence ordering, that `hr` runs on
a generated video (a flat colour frame has no face, so the expected result is a clean
"No face found" error, not a crash), and that every module is 200 lines or fewer.

---

## Project layout

```
deepfake-detector/
├── src/rppg_detector/        the package (every module is 200 lines or fewer)
│   ├── video.py  face.py  signals.py  heartrate.py
│   ├── coherence.py  classify.py  dataset.py  evaluate.py  cli.py
├── tests/                    pytest suite, one file per module
├── scripts/
│   ├── data/                 download, extract, profile and pair up video datasets
│   └── experiments/          run the evaluations and write results/
├── docs/design/              design notes written during the first code generation (may be stale)
├── results/                  one folder per experiment run (see below)
├── data/                     datasets and model files, NOT in git (see Datasets)
├── logs/                     run logs, NOT in git
├── check.sh                  end-to-end success-criteria check
├── prompt.md                 the original project brief
├── pyproject.toml
└── .gitignore
```

Each folder under `results/` holds, for one run: `videos.csv` (per-video fps, size, estimated
heart rate, quality), `windows.csv` (the features of every window with its out-of-fold prediction),
`cv_predictions.csv`, `metrics.json`, `summary.md` and `config.json` (settings and library versions).
The cached skin signals (`signals/`) and trained models (`*.joblib`) are left out of git.

---

## Datasets

Nothing is downloaded by the package or the tests. Everything below is fetched by hand or by the
scripts in `scripts/data/`, into `data/`, which is git-ignored. **Respect each dataset's licence.**

| Dataset | Use | Notes |
|---|---|---|
| [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) | Heart-rate accuracy. Real faces with a pulse-oximeter reference. | Public Google Drive folder; downloads can hit Google's quota and fail for up to a day. `DATASET_2/subjectN/ground_truth.txt` has three rows: PPG waveform, heart rate in BPM, timestamps in seconds. **Not yet used for a full evaluation.** |
| [SDFVD](https://data.mendeley.com/datasets/bcmkfgct2s/1) | Smoke test with face-swap fakes. 53 real clips `vN.mp4` and 53 fakes `vsN.mp4`; `vsN` is the face-swapped version of `vN`. | Clips are only 2 to 6 s, so a pulse cannot really be read. Unzip into `data/sdfvd/`. |
| [GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench) (CC BY-NC 4.0) | Fully AI-generated video (Kling, ModelScope, Pika, SVD, Sora) against real YouTube footage (Vript). | Ungated but 216 GB in `.rar`/`.7z` archives. The scripts download about 67 GB and extract a sample. |
| UADFV | Face swaps with long-enough clips. | Gated on Hugging Face (needs approval); not used. |

---

## Reproducing the experiments

All commands run from the repository root with the environment active and `RPPG_FACE_MODEL` set.

**SDFVD (face swaps, paired by person)**
```bash
python scripts/data/make_sdfvd_manifest.py data/sdfvd/SDFVD data/sdfvd/manifest.csv
python scripts/experiments/run_experiment.py data/sdfvd/manifest.csv --window 2 --step 1 --name sdfvd
```

**GenVidBench (AI-generated video)**
```bash
python scripts/data/download_genvidbench.py                 # about 67 GB, resumable
python scripts/data/extract_genvidbench_subset.py data/genvidbench/GenVidBench data/genvidbench/subset
python scripts/data/profile_clips.py data/genvidbench/subset data/genvidbench/subset_profile.csv

# Every usable Kling clip against real clips matched on fps, duration, encoding and face size
python scripts/data/prepare_kling_full.py data/genvidbench/subset data/genvidbench/subset_profile.csv data/genvidbench/kling_full 2
python scripts/experiments/run_experiment.py data/genvidbench/kling_full/manifest.csv --window 3 --step 1 --name kling_full
python scripts/experiments/nuisance_check.py results/kling_full_<timestamp> data/genvidbench/kling_full/manifest.csv

# About 10 matched pairs from each generator
python scripts/data/prepare_matched_models.py data/genvidbench/subset_profile.csv data/genvidbench/matched 10
python scripts/experiments/run_matched_models.py data/genvidbench/matched/manifest.csv
```

`run_experiment.py` records, per video, the fps, duration, frames without a face and the estimated
heart rate, and per window the four features. It cross-validates with `GroupKFold` by `person_id`
(asserting no person is on both sides), reports window and video ROC AUC with a bootstrap interval
and a label-permutation p-value, and trains a final model. `nuisance_check.py` computes simple
non-pulse statistics (brightness, frame-to-frame jitter, share of variance in the pulse band) to see
whether they separate the classes as well as the detector does.

---

## What the experiments found

AUC is 0.5 for no skill and 1.0 for perfect. Intervals and p-values come from bootstrapping and from
shuffling the labels.

| Experiment (folder in `results/`) | Clips | Result |
|---|---|---|
| SDFVD face swaps, 2 s windows (`sdfvd_20261025_215205`) | 101 with a face | Window AUC 0.53 (p = 0.31), video accuracy 0.55. **Chance.** The fake's estimated heart rate matches its source (63.2 vs 64.2 BPM, p = 0.50), as expected when a swap keeps the original pulse. |
| Kling vs Vript, unmatched (`genvid_kling_vs_vript_20261025_214938`) | 51 fake, 152 real | Window AUC 0.66 (p = 0.005), but a plain variance statistic scored 0.76. Looked promising; most of it was fps, length and compression. |
| Four generators, matched pairs (`matched_models_20261025_220127`) | 41 pairs | Pooled AUC 0.63 (95% interval 0.51 to 0.75, p = 0.035), accuracy 58.5%. About 10 pairs per generator, so per-generator numbers are anecdotes. Sora had only one usable pair. |
| **Full Kling, matched** (`kling_full_20261025_224440`) | 35 fake, 69 real | Video AUC 0.64 (interval 0.54 to 0.73), window p = 0.040. Real and fake heart rates equal (60.0 vs 61.7 BPM, p = 0.41). **A simple jitter measure beats the detector:** share of variance in the pulse band AUC 0.78, band power 0.76, frame-to-frame jitter 0.73. |

**Reading it.** The detector is slightly better than a coin flip on Kling clips, but generated video
is simply smoother than real camera footage, and "how much does the face colour wobble" separates
them better than any pulse feature does. Estimated heart rates are the same for real and fake and
noisy within a pair. The honest conclusion: **no evidence yet that this detector notices a missing
heartbeat.** Stale or superseded runs are also kept in `results/` (`..._214038` crashed,
`..._214641` and `sdfvd_20261025_200437` predate the classifier fix).

**Not yet measured:** heart-rate accuracy against a true pulse. The synthetic 72 BPM test passes, but
nobody has shown the estimator reads a real person's pulse correctly. Until it does (UBFC-rPPG is the
planned test), no downstream result can be interpreted as pulse detection.

---

## Known limits

- **Compressed or moving video hurts the pulse signal.** The signal is a fraction of a percent of
  pixel brightness; compression noise, head motion and lighting changes are larger.
- **Clips shorter than 10 seconds give coarse heart rates.** The FFT resolution is 1 / duration:
  a 3 s clip resolves about 20 BPM. Very short windows also make coherence unreliable, because it
  needs several averaging segments. Most clips in the experiments above are 2 to 5 s.
- **Skin tone affects heart-rate readers.** Darker skin reflects less light, so the signal is weaker.
  This project did not test performance across skin tones.
- **Good deepfakes can copy the pulse from the source video.** A face swap keeps the original
  person's colour changes, so a pulse alone is not proof; SDFVD shows exactly this.
- **A homemade detector will score well below FakeCatcher's reported 96 percent.**
- **Generated video can fake a pulse, and real video can lack one.** Flicker, noise and compression
  all land in the same frequency band as a heartbeat.
- **Many generated clips contain no usable face.** Most prompts are scenes, so only a minority of
  AI-generated clips can be analysed at all.
- **Small samples.** Even the largest test here has 35 fake clips; per-generator comparisons are noise.
- **Landmark patches.** Patches are fixed landmark outlines; hair, glasses, beards, open mouths and
  fast head turns put non-skin pixels in them.

---

## Differences from the original brief

- `dataset.py` was added so `train` can compute real features from a manifest; `evaluate.py` no
  longer contains the placeholder that returned all-zero features.
- The window **quality** feature is the share of variance shared by the patches, not the FFT
  peak-to-band power ratio used by `estimate_heart_rate` (that one is reported by `hr`).
- Heart-rate MAE (`evaluate_heart_rate_accuracy`) takes two arrays of BPM values; there is no loader
  yet for a CSV of reference pulses or for UBFC's `ground_truth.txt`.
- The spec's "dependencies pinned" is implemented as bounded ranges, not exact versions.
- `docs/design/` was generated before the code was debugged and does not describe the final code.

---

## Ideas for improving it

Roughly in order of value:

1. **Validate the heart-rate estimator on UBFC-rPPG** and report its error in BPM.
2. **Implement POS or CHROM** (the `extend_signal_methods` hook), which remove brightness and motion
   far better than the raw green channel.
3. **Better skin regions:** smooth the landmarks over time, mask eyes, mouth and hair, use more patches.
4. **Longer windows and richer features:** peak signal-to-noise ratio and harmonics, heart-rate
   stability over time, phase agreement between patches.
5. **A fairer test:** 50 or more matched pairs per generator at 5 s or longer, cross-generator
   evaluation, and always compare against the simple jitter baseline.
6. **Combine with artefact-based detectors**, since a pulse alone cannot catch pulse-preserving fakes.

---

## Acknowledgements and data licences

The first version of the code was generated with MetaGPT from `prompt.md`, then debugged and extended.
rPPG pipeline ideas follow FakeCatcher (Ciftci et al.) and the UBFC-rPPG work of Bobbia, Macwan,
Benezeth, Mansouri and Dubois. Datasets are used under their own terms: GenVidBench is
CC BY-NC 4.0; SDFVD and UBFC-rPPG are for research use. This repository has no licence file yet.
