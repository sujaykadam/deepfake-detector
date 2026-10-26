"""Signal extraction from video frames."""

import numpy as np
from typing import Tuple, Optional
from rppg_detector.face import FaceDetector, SKIN_PATCHES
from rppg_detector.video import read_video_frames


def extract_rgb_signals(frames: np.ndarray, face_detector: FaceDetector) -> np.ndarray:
    """
    Extract RGB signals from video frames using face detection.
    
    Args:
        frames: Video frames array of shape (num_frames, height, width, 3)
        face_detector: FaceDetector instance for face detection
        
    Returns:
        Array of shape (num_frames, num_patches, 3) with mean RGB values
    """
    num_frames = frames.shape[0]
    num_patches = len(SKIN_PATCHES)
    signals = np.zeros((num_frames, num_patches, 3))
    
    first_face = None
    for i in range(num_frames):
        patches, _ = face_detector.process_frame(frames[i])
        if patches is not None:
            signals[i] = patches
            if first_face is None:
                first_face = i
        elif i > 0:
            # If no face detected, carry forward previous values
            signals[i] = signals[i-1]
    
    # Frames before the first face take that first face's values
    if first_face:
        signals[:first_face] = signals[first_face]
    
    return signals


def get_green_channel(signals: np.ndarray) -> np.ndarray:
    """
    Extract green channel signals from RGB signals.
    
    Args:
        signals: RGB signals array of shape (frames, patches, 3)
        
    Returns:
        Green channel signals array of shape (frames, patches)
    """
    return signals[:, :, 1]  # Green channel is index 1


def compute_mean_rgb_per_patch(frames: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Compute mean RGB values for each skin patch in a single frame.
    
    Args:
        frames: Single frame array of shape (height, width, 3)
        landmarks: Face landmarks array of shape (478, 2)
        
    Returns:
        Array of shape (patches, 3) with mean RGB values
    """
    if landmarks is None:
        return None
    
    patches = []
    for patch_landmarks in SKIN_PATCHES:
        coords = landmarks[patch_landmarks]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frames.shape[1], x_max)
        y_max = min(frames.shape[0], y_max)
        
        patch_pixels = frames[y_min:y_max, x_min:x_max]
        
        if patch_pixels.size > 0:
            mean_rgb = np.mean(patch_pixels, axis=(0, 1))
        else:
            mean_rgb = np.array([0.0, 0.0, 0.0])
        
        patches.append(mean_rgb)
    
    return np.array(patches)


def extend_signal_methods(signals: np.ndarray, method: str = "green") -> np.ndarray:
    """
    Extend signal extraction methods for future implementations.
    
    Args:
        signals: RGB signals array of shape (frames, patches, 3)
        method: Signal extraction method ("green", "chrom", "pos")
        
    Returns:
        Processed signals based on the specified method
    """
    if method == "green":
        return get_green_channel(signals)
    elif method == "chrom":
        # CHROM method: linear combination of R and G channels
        # TODO: Implement CHROM method when needed
        raise NotImplementedError("CHROM method not yet implemented")
    elif method == "pos":
        # POS method: linear combinations of RGB channels
        # TODO: Implement POS method when needed
        raise NotImplementedError("POS method not yet implemented")
    else:
        raise ValueError(f"Unknown signal method: {method}")


def normalize_signals(signals: np.ndarray) -> np.ndarray:
    """
    Normalize signals by subtracting mean and dividing by standard deviation.
    
    Args:
        signals: Signals array of shape (frames, patches)
        
    Returns:
        Normalized signals array of same shape
    """
    means = np.mean(signals, axis=0, keepdims=True)
    stds = np.std(signals, axis=0, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    return (signals - means) / stds


def video_to_green_signals(path: str, face_detector: FaceDetector) -> Tuple[np.ndarray, float]:
    """
    Read a video and return the green-channel skin signal of each patch.
    
    Args:
        path: Path to the video file
        face_detector: FaceDetector instance for face detection
        
    Returns:
        Tuple of (signals, fps) where signals has shape (frames, patches)
        and fps is the frame rate in frames per second (Hz)
        
    Raises:
        ValueError: If no face is found in any frame
    """
    frames, fps = read_video_frames(path)
    rgb = extract_rgb_signals(frames, face_detector)
    if not rgb.any():
        raise ValueError(f"No face found in any frame of {path}")
    return get_green_channel(rgb), fps
