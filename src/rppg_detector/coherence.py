"""Signal coherence analysis for rPPG-based deepfake detection."""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import coherence, butter, sosfiltfilt
from typing import Tuple, List
from rppg_detector.heartrate import detrend, bandpass_filter, estimate_heart_rate_per_patch


def compute_coherence_matrix(signals: np.ndarray, fs: float, 
                            freq_range: Tuple[float, float] = (0.7, 3.0)) -> np.ndarray:
    """
    Compute coherence matrix between all patch signals.
    
    Args:
        signals: Signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        freq_range: Frequency range for coherence analysis (Hz)
        
    Returns:
        Coherence matrix of shape (patches, patches)
    """
    num_patches = signals.shape[1]
    coherence_matrix = np.zeros((num_patches, num_patches))
    
    # Detrend signals
    detrended_signals = detrend(signals)
    
    # Apply bandpass filter
    filtered_signals = bandpass_filter(detrended_signals, fs, 
                                       freq_range[0], freq_range[1])
    
    # Compute coherence between all pairs
    for i in range(num_patches):
        for j in range(i, num_patches):
            with np.errstate(invalid='ignore', divide='ignore'):
                f, coh = coherence(filtered_signals[:, i], 
                                   filtered_signals[:, j], 
                                   fs=fs, nperseg=min(int(4 * fs), len(signals) // 2))
            # A constant signal has no power, so coherence is 0/0: treat as no coupling
            coh = np.nan_to_num(coh, nan=0.0)
            
            # Average coherence in heart rate band
            freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
            avg_coh = np.mean(coh[freq_mask]) if np.any(freq_mask) else 0.0
            
            coherence_matrix[i, j] = avg_coh
            coherence_matrix[j, i] = avg_coh  # Symmetric
    
    return coherence_matrix


def extract_window_features(signals: np.ndarray, fs: float, 
                           window_seconds: float = 10.0, 
                           step_seconds: float = 5.0,
                           freq_range: Tuple[float, float] = (0.7, 3.0)) -> np.ndarray:
    """
    Extract features from sliding windows for classification.
    
    Args:
        signals: Signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        window_seconds: Window size in seconds (default: 10)
        step_seconds: Step size in seconds (default: 5)
        freq_range: Frequency range for analysis (Hz)
        
    Returns:
        Feature vectors array of shape (num_windows, 4), with no rows for
        windows shorter than half the window length, containing:
        - mean_pairwise_coh: Mean pairwise coherence
        - min_pairwise_coh: Minimum pairwise coherence
        - hr_spread: Spread of heart rate estimates across patches
        - quality_score: Heart rate quality measure
    """
    # Convert window sizes to frames
    window_frames = int(window_seconds * fs)
    step_frames = int(step_seconds * fs)
    
    num_frames, num_patches = signals.shape
    
    # Calculate number of windows
    num_windows = max(1, (num_frames - window_frames) // step_frames + 1)
    
    rows = []
    
    for i in range(num_windows):
        # Extract window
        start_frame = i * step_frames
        end_frame = min(start_frame + window_frames, num_frames)
        window_signals = signals[start_frame:end_frame]
        
        if len(window_signals) < window_frames // 2:
            # Skip too short windows
            continue
        
        # Compute coherence matrix for this window
        coh_matrix = compute_coherence_matrix(window_signals, fs, freq_range)
        
        # Extract coherence features (upper triangle, excluding diagonal)
        upper_tri = coh_matrix[np.triu_indices(num_patches, k=1)]
        
        mean_coh = np.mean(upper_tri)
        min_coh = np.min(upper_tri) if len(upper_tri) > 0 else 0.0
        
        # Estimate heart rates for each patch
        hr_values = estimate_heart_rate_per_patch(window_signals, fs)
        hr_spread = np.max(hr_values) - np.min(hr_values) if len(hr_values) > 1 else 0.0
        
        # Compute quality score
        detrended = detrend(window_signals)
        filtered = bandpass_filter(detrended, fs, freq_range[0], freq_range[1])
        
        # Simple quality measure based on signal variance
        total_var = np.sum(np.var(filtered, axis=0))
        signal_var = np.var(np.mean(filtered, axis=1))
        quality_score = signal_var / total_var if total_var > 0 else 0.0
        
        rows.append([mean_coh, min_coh, hr_spread, quality_score])
    
    return np.array(rows).reshape(-1, 4)


def get_window_boundaries(num_frames: int, fs: float, 
                         window_seconds: float = 10.0, 
                         step_seconds: float = 5.0) -> List[Tuple[int, int]]:
    """
    Get window boundaries for sliding window analysis.
    
    Args:
        num_frames: Total number of frames
        fs: Sampling frequency in Hz
        window_seconds: Window size in seconds (default: 10)
        step_seconds: Step size in seconds (default: 5)
        
    Returns:
        List of (start_frame, end_frame) tuples for each window
    """
    window_frames = int(window_seconds * fs)
    step_frames = int(step_seconds * fs)
    
    boundaries = []
    start_frame = 0
    
    while start_frame < num_frames:
        end_frame = min(start_frame + window_frames, num_frames)
        boundaries.append((start_frame, end_frame))
        start_frame += step_frames
    
    return boundaries
