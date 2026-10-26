"""Tests for signals.py module."""

import numpy as np
import pytest
from rppg_detector.signals import extract_rgb_signals, get_green_channel, \
    compute_mean_rgb_per_patch, extend_signal_methods, normalize_signals
from rppg_detector.face import FaceDetector


def create_stub_face_detector():
    """Create a stub face detector for testing."""
    class StubFaceDetector(FaceDetector):
        def detect_faces(self, frame):
            # Return some landmarks for testing
            landmarks = np.random.rand(478, 2) * 100
            return landmarks
        
        def get_skin_patches(self, frame, landmarks):
            # Return fixed RGB values for testing
            return np.array([[50.0, 100.0, 150.0],  # Forehead
                           [55.0, 105.0, 145.0],  # Left cheek
                           [48.0, 98.0, 152.0]])  # Right cheek
    
    return StubFaceDetector()


def test_extract_rgb_signals():
    """Test RGB signal extraction."""
    # Create test frames
    frames = np.zeros((30, 480, 640, 3), dtype=np.uint8)  # 30 frames
    frames[:] = [50, 100, 150]  # Skin tone color
    
    # Create face detector
    face_detector = create_stub_face_detector()
    
    # Extract signals
    signals = extract_rgb_signals(frames, face_detector)
    
    # Verify results
    assert signals.shape == (30, 3, 3)  # 30 frames, 3 patches, 3 color channels
    assert np.all(signals[0] == np.array([[50.0, 100.0, 150.0], 
                                         [55.0, 105.0, 145.0], 
                                         [48.0, 98.0, 152.0]]))


def test_get_green_channel():
    """Test green channel extraction."""
    # Create test signals
    signals = np.array([[[50.0, 100.0, 150.0],  # Frame 1
                        [55.0, 105.0, 145.0],
                        [48.0, 98.0, 152.0]],
                       [[52.0, 102.0, 148.0],  # Frame 2
                        [57.0, 107.0, 143.0],
                        [50.0, 100.0, 150.0]]])
    
    # Extract green channel
    green = get_green_channel(signals)
    
    # Verify results
    assert green.shape == (2, 3)  # 2 frames, 3 patches
    assert np.allclose(green[0], [100.0, 105.0, 98.0])
    assert np.allclose(green[1], [102.0, 107.0, 100.0])


def test_compute_mean_rgb_per_patch():
    """Test mean RGB computation for patches."""
    # Create test landmarks
    landmarks = np.tile([[100, 100], [120, 120]], (239, 1))[:478]
    
    # Create test frame with known values
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[100:120, 100:120] = [50, 100, 150]  # Region with known values
    
    # Compute mean RGB
    patches = compute_mean_rgb_per_patch(frame, landmarks)
    
    # Verify results
    assert patches is not None
    assert patches.shape == (3, 3)  # 3 patches, 3 color channels
    # Note: This test may vary based on landmark selection


def test_normalize_signals():
    """Test signal normalization."""
    # Create test signals
    signals = np.array([[100.0, 105.0, 102.0],
                       [102.0, 107.0, 104.0],
                       [98.0, 103.0, 100.0]])
    
    # Normalize
    normalized = normalize_signals(signals)
    
    # Verify results
    assert normalized.shape == signals.shape
    # Check that mean is approximately zero
    assert np.allclose(np.mean(normalized, axis=0), [0.0, 0.0, 0.0], atol=1e-10)
    # Check that standard deviation is approximately 1
    assert np.allclose(np.std(normalized, axis=0), [1.0, 1.0, 1.0], atol=1e-10)


def test_extend_signal_methods_green():
    """Test green channel signal extraction."""
    # Create test signals
    signals = np.array([[[50.0, 100.0, 150.0],  # Frame 1
                        [55.0, 105.0, 145.0]],
                       [[52.0, 102.0, 148.0],  # Frame 2
                        [57.0, 107.0, 143.0]]])
    
    # Extract using extend_signal_methods
    green = extend_signal_methods(signals, method="green")
    
    # Verify results
    assert green.shape == (2, 2)  # 2 frames, 2 patches
    assert np.allclose(green[0], [100.0, 105.0])


def test_extend_signal_methods_unsupported():
    """Test unsupported signal method."""
    signals = np.zeros((10, 3, 3))
    
    with pytest.raises(ValueError):
        extend_signal_methods(signals, method="unsupported")


def test_signal_shapes():
    """Test that signal arrays have correct shapes."""
    # Create test frames
    frames = np.zeros((20, 480, 640, 3), dtype=np.uint8)
    face_detector = create_stub_face_detector()
    
    # Extract signals
    signals = extract_rgb_signals(frames, face_detector)
    
    # Verify shape
    assert len(signals.shape) == 3  # (frames, patches, channels)
    assert signals.shape[2] == 3   # RGB channels


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
