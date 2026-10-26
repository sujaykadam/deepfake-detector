"""Heart rate estimation from physiological signals."""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, sosfiltfilt
from typing import Tuple


def detrend(signals: np.ndarray, lambda_val: float = 100) -> np.ndarray:
    """
    Detrend signals to remove slow drift.
    
    Args:
        signals: Signals array of shape (frames, patches)
        lambda_val: Regularization parameter for detrending
        
    Returns:
        Detrended signals array of same shape
    """
    # Simple detrending using scipy's detrend function
    return scipy_signal.detrend(signals, axis=0)


def bandpass_filter(signals: np.ndarray, fs: float, low_freq: float = 0.7, 
                   high_freq: float = 3.0) -> np.ndarray:
    """
    Apply bandpass filter to signals.
    
    Args:
        signals: Signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        low_freq: Lower cutoff frequency in Hz (default: 0.7)
        high_freq: Upper cutoff frequency in Hz (default: 3.0)
        
    Returns:
        Filtered signals array of same shape
    """
    sos = butter(4, [low_freq, high_freq], btype='bandpass', fs=fs, output='sos')
    # Very short clips (a second or two) are shorter than the default edge padding
    padlen = min(3 * (2 * sos.shape[0] + 1), len(signals) - 1)
    return sosfiltfilt(sos, signals, axis=0, padlen=padlen)


def estimate_heart_rate(signals: np.ndarray, fs: float, 
                       freq_range: Tuple[float, float] = (0.7, 3.0)) -> Tuple[float, float]:
    """
    Estimate heart rate from signals using FFT.
    
    Args:
        signals: Signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        freq_range: Frequency range for heart rate estimation (Hz)
        
    Returns:
        Tuple of (heart_rate_bpm, quality_score) where:
        - heart_rate_bpm: Estimated heart rate in beats per minute
        - quality_score: Quality measure (peak power / total power)
    """
    num_frames, num_patches = signals.shape
    
    # Detrend signals
    detrended_signals = detrend(signals)
    
    # Apply bandpass filter
    filtered_signals = bandpass_filter(detrended_signals, fs, 
                                       freq_range[0], freq_range[1])
    
    # Compute FFT for each patch
    fft_results = []
    freqs = np.fft.rfftfreq(num_frames, 1.0 / fs)
    
    for i in range(num_patches):
        patch_signal = filtered_signals[:, i]
        fft = np.fft.rfft(patch_signal)
        fft_magnitude = np.abs(fft)
        fft_results.append(fft_magnitude)
    
    # Combine FFT results across patches
    combined_fft = np.mean(fft_results, axis=0)
    
    # Find frequency within heart rate band with maximum power
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    band_power = combined_fft * freq_mask
    
    if np.sum(freq_mask) == 0:
        return 0.0, 0.0
    
    # Find peak frequency
    peak_idx = np.argmax(band_power)
    peak_freq = freqs[peak_idx]
    
    # Convert to BPM
    heart_rate_bpm = peak_freq * 60.0
    
    # Calculate quality score
    total_power = np.sum(combined_fft[freq_mask])
    peak_power = combined_fft[peak_idx]
    
    if total_power > 0:
        quality_score = peak_power / total_power
    else:
        quality_score = 0.0
    
    return heart_rate_bpm, quality_score


def estimate_heart_rate_per_patch(signals: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate heart rate for each patch separately.
    
    Args:
        signals: Signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        
    Returns:
        Array of heart rates in BPM for each patch
    """
    _, num_patches = signals.shape
    heart_rates = np.zeros(num_patches)
    
    for i in range(num_patches):
        patch_signal = signals[:, i].reshape(-1, 1)
        hr, _ = estimate_heart_rate(patch_signal, fs)
        heart_rates[i] = hr
    
    return heart_rates
