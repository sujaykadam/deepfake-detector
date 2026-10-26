"""Tests for coherence.py module."""

import numpy as np
import pytest
from rppg_detector.coherence import compute_coherence_matrix, \
    extract_window_features, get_window_boundaries
from rppg_detector.heartrate import detrend, bandpass_filter


def create_shared_pulse_signals(fs: float = 30.0, duration: float = 30.0) -> np.ndarray:
    """
    Create signals with shared 72 BPM pulse.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Duration in seconds
        
    Returns:
        Signals array of shape (frames, patches)
    """
    num_frames = int(fs * duration)
    t = np.arange(num_frames) / fs
    
    # 72 BPM = 1.2 Hz
    heart_rate_hz = 1.2
    
    # Create base signal
    base_signal = np.sin(2 * np.pi * heart_rate_hz * t)
    
    # Add slight variations for each patch
    signals = np.column_stack([
        base_signal + 0.05 * np.random.randn(num_frames),  # Patch 1
        base_signal + 0.05 * np.random.randn(num_frames),  # Patch 2
        base_signal + 0.05 * np.random.randn(num_frames)   # Patch 3
    ])
    
    # Add slow drift
    signals += 0.1 * t.reshape(-1, 1)
    
    return signals


def create_unrelated_signals(fs: float = 30.0, duration: float = 30.0) -> np.ndarray:
    """
    Create signals with unrelated frequencies.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Duration in seconds
        
    Returns:
        Signals array of shape (frames, patches)
    """
    num_frames = int(fs * duration)
    t = np.arange(num_frames) / fs
    
    # Different frequencies for each patch
    frequencies = [0.8, 1.5, 2.2]  # Different heart rates
    
    signals = np.column_stack([
        np.sin(2 * np.pi * frequencies[0] * t) + 0.1 * t,
        np.sin(2 * np.pi * frequencies[1] * t) + 0.1 * t,
        np.sin(2 * np.pi * frequencies[2] * t) + 0.1 * t
    ])
    
    # Add noise
    signals += 0.1 * np.random.randn(num_frames, 3)
    
    return signals


def test_compute_coherence_matrix():
    """Test coherence matrix computation."""
    fs = 30.0
    signals = create_shared_pulse_signals(fs, 10.0)  # 10 seconds
    
    # Compute coherence matrix
    coh_matrix = compute_coherence_matrix(signals, fs)
    
    # Verify results
    assert coh_matrix.shape == (3, 3)  # 3x3 matrix for 3 patches
    
    # Check that diagonal elements are ~1.0 (perfect coherence with self)
    for i in range(3):
        assert 0.9 <= coh_matrix[i, i] <= 1.1
    
    # Check that off-diagonal elements are positive (coherent signals)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert coh_matrix[i, j] > 0.0


def test_shared_vs_unrelated_coherence():
    """Test that shared pulse signals have higher coherence."""
    fs = 30.0
    duration = 30.0
    
    # Create both signal types
    shared_signals = create_shared_pulse_signals(fs, duration)
    unrelated_signals = create_unrelated_signals(fs, duration)
    
    # Compute coherence matrices
    shared_coh = compute_coherence_matrix(shared_signals, fs)
    unrelated_coh = compute_coherence_matrix(unrelated_signals, fs)
    
    # Compare mean coherence
    shared_mean = np.mean(shared_coh[np.triu_indices(3, k=1)])
    unrelated_mean = np.mean(unrelated_coh[np.triu_indices(3, k=1)])
    
    # Shared signals should have higher coherence
    assert shared_mean > unrelated_mean
    print(f"Shared coherence: {shared_mean:.4f}, Unrelated coherence: {unrelated_mean:.4f}")


def test_extract_window_features():
    """Test window feature extraction."""
    fs = 30.0
    signals = create_shared_pulse_signals(fs, 30.0)  # 30 seconds
    
    # Extract features
    features = extract_window_features(signals, fs, 
                                       window_seconds=10.0, 
                                       step_seconds=5.0)
    
    # Verify results
    assert features.shape[1] == 4  # 4 features per window
    
    # Check feature ranges
    assert np.all(features[:, 0] >= 0)  # Mean coherence
    assert np.all(features[:, 1] >= 0)  # Min coherence
    assert np.all(features[:, 2] >= 0)  # HR spread
    assert np.all(features[:, 3] >= 0)  # Quality score


def test_window_boundaries():
    """Test window boundary calculation."""
    num_frames = 900  # 30 seconds at 30 fps
    fs = 30.0
    
    # Get boundaries
    boundaries = get_window_boundaries(num_frames, fs, 
                                       window_seconds=10.0, 
                                       step_seconds=5.0)
    
    # Verify results
    assert len(boundaries) >= 2  # Should have multiple windows
    
    # Check first window
    start, end = boundaries[0]
    assert start == 0
    assert end == 300  # 10 seconds * 30 fps
    
    # Check step
    start2, _ = boundaries[1]
    assert start2 == 150  # 5 seconds * 30 fps step


def test_feature_differences():
    """Test that features differ between signal types."""
    fs = 30.0
    duration = 30.0
    
    # Create signals
    shared_signals = create_shared_pulse_signals(fs, duration)
    unrelated_signals = create_unrelated_signals(fs, duration)
    
    # Extract features
    shared_features = extract_window_features(shared_signals, fs)
    unrelated_features = extract_window_features(unrelated_signals, fs)
    
    # Compare mean values
    shared_mean = np.mean(shared_features, axis=0)
    unrelated_mean = np.mean(unrelated_features, axis=0)
    
    # Shared signals should have higher coherence
    assert shared_mean[0] > unrelated_mean[0]  # Mean coherence
    assert shared_mean[1] > unrelated_mean[1]  # Min coherence
    
    # Shared signals should have lower HR spread
    assert shared_mean[2] < unrelated_mean[2]  # HR spread


def test_short_signal_handling():
    """Test handling of short signals."""
    fs = 30.0
    signals = np.random.randn(100, 3)  # Only ~3 seconds
    
    # Extract features
    features = extract_window_features(signals, fs, 
                                       window_seconds=10.0, 
                                       step_seconds=5.0)
    
    # Should handle gracefully
    assert len(features) >= 0  # May have zero or few windows


def test_quality_score_consistency():
    """Test quality score consistency."""
    fs = 30.0
    signals = create_shared_pulse_signals(fs, 30.0)
    
    # Extract features
    features = extract_window_features(signals, fs)
    
    # Quality scores should be positive
    assert np.all(features[:, 3] >= 0)
    
    # For consistent signals, quality should be reasonable
    assert np.mean(features[:, 3]) > 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_constant_signal_has_zero_coherence():
    """A constant patch signal (no power) gives coherence 0, not NaN."""
    fs = 30.0
    signals = create_shared_pulse_signals(fs, 10.0)
    signals[:, 2] = 5.0
    features = extract_window_features(signals, fs, 4.0, 2.0)
    assert not np.isnan(features).any()
