"""Tests for heartrate.py module."""

import numpy as np
import pytest
from rppg_detector.heartrate import detrend, bandpass_filter, \
    estimate_heart_rate, estimate_heart_rate_per_patch


def create_72_bpm_signal(fs: float = 30.0, duration: float = 30.0) -> np.ndarray:
    """
    Create a synthetic 72 BPM signal.
    
    Args:
        fs: Sampling frequency in Hz
        duration: Duration in seconds
        
    Returns:
        Signal array
    """
    num_frames = int(fs * duration)
    t = np.arange(num_frames) / fs
    
    # 72 BPM = 1.2 Hz
    heart_rate_hz = 1.2
    
    # Create signal: sine wave + drift + noise
    signal = np.sin(2 * np.pi * heart_rate_hz * t)
    signal += 0.1 * t  # Slow drift
    signal += 0.05 * np.random.randn(num_frames)  # Noise
    
    return signal.reshape(-1, 1)


def test_detrend():
    """Test signal detrending."""
    # Create signal with trend
    signal = np.arange(100).reshape(-1, 1) + np.random.randn(100, 1) * 0.1
    
    # Detrend
    detrended = detrend(signal)
    
    # Verify detrending worked
    assert detrended.shape == signal.shape
    # Check that mean is approximately zero after detrending
    assert np.abs(np.mean(detrended)) < 0.1


def test_bandpass_filter():
    """Test bandpass filtering."""
    fs = 30.0
    num_frames = 900  # 30 seconds at 30 fps
    t = np.arange(num_frames) / fs
    
    # Create signal with multiple frequencies
    signal = (np.sin(2 * np.pi * 0.5 * t) +  # Below passband
              np.sin(2 * np.pi * 1.5 * t) +  # In passband (0.7-3.0 Hz)
              np.sin(2 * np.pi * 4.0 * t))   # Above passband
    
    signal = signal.reshape(-1, 1)
    
    # Apply filter
    filtered = bandpass_filter(signal, fs, 0.7, 3.0)
    
    # Verify filtering worked
    assert filtered.shape == signal.shape
    # Check that low frequency component is reduced
    # (this is a qualitative test)


def test_estimate_72_bpm_signal():
    """Test heart rate estimation on 72 BPM signal."""
    fs = 30.0
    signal = create_72_bpm_signal(fs, 30.0)
    
    # Estimate heart rate
    hr, quality = estimate_heart_rate(signal, fs)
    
    # Verify results (72 BPM = 1.2 Hz)
    assert abs(hr - 72.0) <= 3.0  # Within 3 BPM
    assert quality > 0.1  # Should have reasonable quality
    print(f"Estimated HR: {hr:.1f} BPM, Quality: {quality:.4f}")


def test_estimate_heart_rate_per_patch():
    """Test per-patch heart rate estimation."""
    fs = 30.0
    num_frames = 900  # 30 seconds
    
    # Create signals for 3 patches, all at 72 BPM
    t = np.arange(num_frames) / fs
    signal_1 = np.sin(2 * np.pi * 1.2 * t) + 0.1 * t
    signal_2 = np.sin(2 * np.pi * 1.2 * t + 0.1) + 0.1 * t  # Slightly phase-shifted
    signal_3 = np.sin(2 * np.pi * 1.2 * t - 0.1) + 0.1 * t  # Slightly phase-shifted
    
    signals = np.column_stack([signal_1, signal_2, signal_3])
    
    # Estimate heart rates
    heart_rates = estimate_heart_rate_per_patch(signals, fs)
    
    # Verify results
    assert len(heart_rates) == 3
    assert np.allclose(heart_rates, 72.0, atol=3.0)  # All within 3 BPM of 72


def test_quality_score():
    """Test quality score calculation."""
    fs = 30.0
    signal = create_72_bpm_signal(fs, 30.0)
    
    # Estimate heart rate
    hr, quality = estimate_heart_rate(signal, fs)
    
    # Verify quality score is reasonable
    assert 0.0 <= quality <= 1.0
    
    # Test with noisy signal
    noisy_signal = signal + 0.5 * np.random.randn(len(signal), 1)
    _, noisy_quality = estimate_heart_rate(noisy_signal, fs)
    
    # Noisy signal should have lower quality
    assert noisy_quality < quality


def test_different_frequencies():
    """Test heart rate estimation at different frequencies."""
    fs = 30.0
    num_frames = 900
    t = np.arange(num_frames) / fs
    
    # Test various heart rates
    test_rates = [60, 72, 90, 120]  # BPM
    
    for rate in test_rates:
        freq = rate / 60.0  # Convert to Hz
        signal = (np.sin(2 * np.pi * freq * t) + 
                 0.1 * t + 
                 0.05 * np.random.randn(num_frames))
        
        signal = signal.reshape(-1, 1)
        hr, _ = estimate_heart_rate(signal, fs)
        
        # Verify within 5 BPM
        assert abs(hr - rate) <= 5.0
        print(f"Tested {rate} BPM, estimated {hr:.1f} BPM")


def test_insufficient_data():
    """Test heart rate estimation with insufficient data."""
    fs = 30.0
    signal = np.random.randn(30, 1)  # Only 1 second of data
    
    # Should handle gracefully
    hr, quality = estimate_heart_rate(signal, fs)
    
    # Result should still be returned (may be less accurate)
    assert isinstance(hr, float)
    assert isinstance(quality, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_bandpass_handles_very_short_clip():
    """A 1.4 s clip at 10 fps (14 frames) is filtered without an edge-padding error."""
    from rppg_detector.heartrate import bandpass_filter
    signal = np.random.randn(14, 3)
    assert bandpass_filter(signal, 10.0).shape == (14, 3)
