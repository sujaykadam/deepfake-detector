"""Tests for face.py module."""

import numpy as np
import pytest
from rppg_detector.face import FaceDetector, get_skin_patch_indices, \
    FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS, SKIN_PATCHES


def test_skin_patch_indices():
    """Test that skin patch indices are correctly defined."""
    # Check that all patch lists are non-empty
    assert len(FOREHEAD_LANDMARKS) > 0
    assert len(LEFT_CHEEK_LANDMARKS) > 0
    assert len(RIGHT_CHEEK_LANDMARKS) > 0
    
    # Check that SKIN_PATCHES contains all patches
    assert SKIN_PATCHES[0] == FOREHEAD_LANDMARKS
    assert SKIN_PATCHES[1] == LEFT_CHEEK_LANDMARKS
    assert SKIN_PATCHES[2] == RIGHT_CHEEK_LANDMARKS
    
    # Check that we have exactly 3 patches
    assert len(SKIN_PATCHES) == 3


def test_get_skin_patch_indices():
    """Test the get_skin_patch_indices function."""
    indices = get_skin_patch_indices()
    assert len(indices) == 3
    assert indices[0] == FOREHEAD_LANDMARKS


def test_face_detector_initialization():
    """Test FaceDetector initialization."""
    detector = FaceDetector()
    assert detector._previous_landmarks is None
    assert detector._dropped_frames == 0


def test_face_detector_reset():
    """Test FaceDetector reset functionality."""
    detector = FaceDetector()
    # Manually set some state
    detector._previous_landmarks = np.zeros((478, 2))
    detector._dropped_frames = 5
    
    detector.reset()
    
    assert detector._previous_landmarks is None
    assert detector._dropped_frames == 0


def test_process_frame_with_stub():
    """Test frame processing with a stub face detector."""
    class StubFaceDetector(FaceDetector):
        def detect_faces(self, frame):
            # Return some landmarks for testing
            landmarks = np.random.rand(478, 2) * 100
            return landmarks
    
    detector = StubFaceDetector()
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [50, 100, 150]  # Skin tone color
    
    # Process frame
    patches, is_dropped = detector.process_frame(frame)
    
    # Verify results
    assert patches is not None
    assert patches.shape == (3, 3)  # 3 patches, 3 color channels
    assert not is_dropped


def test_get_skin_patches():
    """Test skin patch extraction."""
    detector = FaceDetector()
    
    # Create test landmarks
    landmarks = np.random.rand(478, 2) * 100
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [50, 100, 150]  # Skin tone color
    
    # Extract patches
    patches = detector.get_skin_patches(frame, landmarks)
    
    # Verify results
    assert patches is not None
    assert patches.shape == (3, 3)  # 3 patches, 3 color channels


def test_process_frame_no_face():
    """Test frame processing when no face is detected."""
    class NoFaceDetector(FaceDetector):
        def detect_faces(self, frame):
            return None

    detector = NoFaceDetector()
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Process frame (no face detected)
    patches, is_dropped = detector.process_frame(frame)
    
    # Verify results
    assert patches is None
    assert is_dropped  # Should be True when no face detected


def test_landmark_count():
    """Test that we have the expected number of landmarks."""
    total_landmarks = len(FOREHEAD_LANDMARKS) + len(LEFT_CHEEK_LANDMARKS) + len(RIGHT_CHEEK_LANDMARKS)
    assert total_landmarks <= 478  # Should not exceed total landmarks


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_missing_model_gives_clear_error(monkeypatch):
    """Test that detecting without a model file explains how to provide one."""
    monkeypatch.delenv("RPPG_FACE_MODEL", raising=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="RPPG_FACE_MODEL"):
        FaceDetector().detect_faces(frame)
