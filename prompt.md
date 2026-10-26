Build a Python package `rppg_detector`: a learning-project deepfake detector based on remote photoplethysmography (rPPG), the idea behind FakeCatcher. Every heartbeat changes how much green light skin reflects, a camera records it, and on a real face the pulse signal from different skin patches agrees; on a fake it agrees less. The package turns a face video into a heart rate and a real-or-fake score.

Stack: Python 3.11, OpenCV, MediaPipe Face Landmarker, numpy, scipy, scikit-learn, pytest. CLI only, no web UI, no GPU, no deep learning. The code must run on a laptop. Do not download datasets inside the code or tests.

## Modules

1. `video.py`: read frames with `cv2.VideoCapture`. Read the real frame rate with `cap.get(cv2.CAP_PROP_FPS)` and return it with the frames, because every beats-per-minute number depends on it. Raise a clear error if the file cannot be read.
2. `face.py`: use MediaPipe Face Landmarker (478 landmarks) to find the face in each frame. Define three skin patches as fixed landmark index lists: forehead, left cheek, right cheek. Keep the index lists in one constants block with a comment saying which landmarks they are. If no face is found in a frame, carry the previous value forward and count it as a dropped frame.
3. `signals.py`: for each patch and frame, compute the mean red, green and blue of the patch pixels. Return an array of shape (frames, patches, 3). Provide the plain green-channel signal as the default method. Leave a clear extension point for CHROM and POS, but do not implement them.
4. `heartrate.py`:
   - `detrend` the signal with `scipy.signal.detrend`.
   - Band-pass 0.7 to 3.0 Hz with `scipy.signal.butter(..., btype='bandpass', fs=fs, output='sos')` and `scipy.signal.sosfiltfilt`.
   - Find the strongest frequency in that band with `numpy.fft.rfft` and `rfftfreq`, multiply by 60, and return beats per minute.
   - Return a quality measure: the strongest peak's power divided by the total power in the band.
5. `coherence.py`: for each sliding window (default 10 seconds, 5 second step), measure how closely the filtered signals of the three patches agree using `scipy.signal.coherence`, evaluated inside the heart-rate band. Return one short feature vector per window: the mean pairwise coherence, the minimum pairwise coherence, the spread of the per-patch heart-rate estimates, and the heart-rate quality measure.
6. `classify.py`: train a logistic regression on window feature vectors labelled real (0) or fake (1). To classify a video, classify every window and let the majority vote decide; also return the fraction of windows that voted fake as a score. Save and load the model with joblib.
7. `evaluate.py`:
   - Heart-rate mean absolute error in beats per minute against a CSV of reference pulses.
   - Detector accuracy using `sklearn.model_selection.GroupKFold`. The groups are person IDs, so every clip of one person, and every fake made from one real video, stay on the same side of the split. Never use a random split.
8. `cli.py`: use argparse with three commands.
   - `rppg_detector hr VIDEO` prints the estimated beats per minute and the quality.
   - `rppg_detector train MANIFEST_CSV --out MODEL` trains the classifier. The CSV columns are `path,label,person_id`.
   - `rppg_detector predict VIDEO --model MODEL` prints real or fake with the score.

## Tests (pytest, no network, no real video files needed)

The tests must generate their own data:
- A synthetic signal: a 1.2 Hz sine wave (72 beats per minute) plus slow drift plus noise, sampled at 30 fps for 30 seconds. The heart-rate function must return 72 beats per minute within 3.
- Three synthetic patch signals sharing the same 1.2 Hz pulse must give high coherence. Three signals with unrelated random frequencies must give clearly lower coherence.
- A synthetic video written with `cv2.VideoWriter`, a flat skin-coloured frame whose green value is modulated at 1.2 Hz. The video reader must report the right frame rate and frame count. Use a stub for the face finder in the test, so MediaPipe is not needed.
- The classifier must train on two clearly separable synthetic feature sets and get at least 90 percent accuracy on a held-out set.
- A test that `GroupKFold` never puts one `person_id` on both sides of a split.

## Success criteria

All of these must hold, and `./check.sh` should check them:
1. `pip install -e .` works from a fresh Python 3.11 virtual environment, with dependencies pinned in `pyproject.toml`.
2. `pytest` passes with at least 15 tests.
3. The synthetic 72 beats per minute test passes within 3 beats per minute.
4. The coherence test shows the shared-pulse signals scoring higher than the unrelated signals.
5. `rppg_detector hr` runs on a generated synthetic video without crashing.
6. No module is longer than 200 lines, and every public function has a docstring that says what its inputs and outputs are, including units (Hz, beats per minute, seconds).
7. The README explains the idea in plain language, shows how to run each CLI command, and lists these known limits: compressed or moving video hurts the pulse signal, clips shorter than 10 seconds give coarse heart rates, skin tone affects heart-rate readers, good deepfakes can copy the pulse from the source video so a pulse alone is not proof, and a homemade detector will score well below FakeCatcher's reported 96 percent.

Keep the code simple and readable. Prefer small functions. Put the frame rate in every function that converts between frequency and time. Do not add features beyond this list.


## Environment rules
- Never run interactive commands or anything that waits for input.
- If a command fails, do not retry it more than twice; try another alternative approach.
