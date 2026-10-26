"""Turn a manifest of labelled videos into window feature vectors."""

import sys
from typing import Tuple

import numpy as np
import pandas as pd

from rppg_detector.coherence import extract_window_features
from rppg_detector.face import FaceDetector
from rppg_detector.signals import video_to_green_signals


def load_manifest_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a manifest CSV file.
    
    Args:
        csv_path: Path to a CSV with columns path, label, person_id
        
    Returns:
        DataFrame with those columns (label: 0 = real, 1 = fake)
    """
    return pd.read_csv(csv_path)


def build_window_dataset(manifest: pd.DataFrame, face_detector: FaceDetector,
                         window_seconds: float = 10.0, step_seconds: float = 5.0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one feature vector per sliding window of every manifest video.
    
    Args:
        manifest: DataFrame with columns path, label (0 real, 1 fake), person_id
        face_detector: FaceDetector used to find the skin patches
        window_seconds: Window length in seconds
        step_seconds: Window step in seconds
        
    Returns:
        Tuple of (X, y, groups): X has shape (windows, 4); y and groups have
        shape (windows,) and repeat each video's label and person_id.
        Videos with no detectable face are skipped with a message on stderr.
    """
    X, y, groups = [], [], []
    for row in manifest.itertuples():
        face_detector.reset()
        try:
            signals, fps = video_to_green_signals(row.path, face_detector)
        except ValueError as error:
            print(f"Skipping {row.path}: {error}", file=sys.stderr)
            continue
        features = extract_window_features(signals, fps, window_seconds, step_seconds)
        X.append(features)
        y += [row.label] * len(features)
        groups += [row.person_id] * len(features)
    return np.vstack(X), np.array(y), np.array(groups)
