"""Tests for evaluate.py module."""

import cv2
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from rppg_detector.evaluate import evaluate_heart_rate_accuracy, \
    cross_validate_classifier, validate_group_kfold, \
    evaluate_detector_on_video
from rppg_detector.dataset import build_window_dataset, load_manifest_csv
from rppg_detector.face import FaceDetector
from rppg_detector.classify import DeepfakeClassifier
from tests.synthetic import create_synthetic_training_data


def test_evaluate_heart_rate_accuracy():
    """Test heart rate accuracy evaluation."""
    # Create test predictions and references
    predictions = np.array([72.0, 75.0, 68.0, 80.0])
    references = np.array([70.0, 74.0, 70.0, 78.0])
    
    # Calculate MAE
    mae = evaluate_heart_rate_accuracy(predictions, references)
    
    # Verify MAE calculation
    expected_mae = np.mean(np.abs(predictions - references))
    assert abs(mae - expected_mae) < 0.001
    
    # Check specific value
    assert mae == 1.75  # (2+1+2+2)/4


def test_cross_validate_classifier():
    """Test cross-validation functionality."""
    # Create synthetic data
    X = np.random.randn(100, 20)
    y = np.array([0] * 50 + [1] * 50)
    groups = np.array([i // 10 for i in range(100)])  # 10 groups
    
    # Run cross-validation
    results = cross_validate_classifier(X, y, groups, n_splits=5)
    
    # Verify results structure
    assert 'train_accuracy' in results
    assert 'test_accuracy' in results
    assert len(results['train_accuracy']) == 5  # 5 splits
    
    # Verify summary statistics
    assert 'summary' in results
    assert 'mean_test_accuracy' in results['summary']


def test_validate_group_kfold():
    """Test GroupKFold validation."""
    # Create groups
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    
    # Validate
    is_valid = validate_group_kfold(groups, n_splits=2)
    
    # Should be valid (each group in exactly one split)
    assert is_valid


def test_validate_group_kfold_invalid():
    """Test GroupKFold with problematic groups."""
    # Create groups with potential overlap issues
    groups = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Validate
    is_valid = validate_group_kfold(groups, n_splits=5)
    
    # Should still be valid (each group in exactly one split)
    assert is_valid


def test_load_manifest_csv():
    """Test manifest CSV loading."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("path,label,person_id\n")
        f.write("video1.mp4,0,1\n")
        f.write("video2.mp4,1,2\n")
        f.write("video3.mp4,0,1\n")
        temp_path = f.name
    
    try:
        # Load CSV
        df = load_manifest_csv(temp_path)
        
        # Verify results
        assert len(df) == 3
        assert 'path' in df.columns
        assert 'label' in df.columns
        assert 'person_id' in df.columns
        assert df.iloc[0]['path'] == 'video1.mp4'
        assert df.iloc[0]['label'] == 0
        assert df.iloc[0]['person_id'] == 1
    finally:
        os.unlink(temp_path)


class StubFaceDetector(FaceDetector):
    """Face detector that returns fixed landmarks, so MediaPipe is not needed."""

    def detect_faces(self, frame):
        rng = np.random.default_rng(0)
        return rng.uniform([60, 40], [260, 200], size=(478, 2))


def write_pulse_video(path, fps=30.0, seconds=12.0):
    """Write a skin-coloured video whose green value pulses at 1.2 Hz."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))
    for i in range(int(fps * seconds)):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = [150, int(100 + 20 * np.sin(2 * np.pi * 1.2 * i / fps)), 50]
        out.write(frame)
    out.release()


def test_build_window_dataset(tmp_path):
    """Test that each manifest video yields windows with its label and person."""
    paths = []
    for name in ("a.mp4", "b.mp4"):
        write_pulse_video(str(tmp_path / name))
        paths.append(str(tmp_path / name))
    manifest = pd.DataFrame({'path': paths, 'label': [0, 1], 'person_id': [1, 2]})

    X, y, groups = build_window_dataset(manifest, StubFaceDetector(),
                                        window_seconds=6.0, step_seconds=3.0)

    assert X.shape[1] == 4
    assert len(X) == len(y) == len(groups) >= 4
    assert set(y) == {0, 1}
    assert set(groups[y == 0]) == {1} and set(groups[y == 1]) == {2}


def test_evaluate_detector_on_video():
    """Test detector evaluation on video."""
    # Create synthetic data
    X, y = create_synthetic_training_data(n_real=100, n_fake=100)
    
    # Train classifier
    classifier = DeepfakeClassifier()
    classifier.train(X, y)
    
    # Create test signals
    fs = 30.0
    signals = np.random.randn(900, 3)  # 30 seconds
    
    # Evaluate
    results = evaluate_detector_on_video(signals, fs, classifier)
    
    # Verify results
    assert 'fake_fraction' in results
    assert 'avg_prob_fake' in results
    assert 'num_windows' in results
    assert 'windows_voted_fake' in results
    assert 'windows_voted_real' in results
    
    assert 0.0 <= results['fake_fraction'] <= 1.0
    assert 0.0 <= results['avg_prob_fake'] <= 1.0


def test_cross_validation_with_realistic_data():
    """Test cross-validation with more realistic data."""
    # Create data with clearer separation
    n_samples = 100
    n_features = 20
    
    # Create features with clear separation
    real_features = np.random.randn(n_samples // 2, n_features) + 2.0
    fake_features = np.random.randn(n_samples // 2, n_features) - 2.0
    
    X = np.vstack([real_features, fake_features])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    groups = np.array([i // 5 for i in range(n_samples)])  # Groups
    
    # Run cross-validation
    results = cross_validate_classifier(X, y, groups, n_splits=5)
    
    # Check accuracy
    mean_acc = results['summary']['mean_test_accuracy']
    assert mean_acc >= 0.80  # Should achieve at least 80% accuracy
    print(f"Realistic CV accuracy: {mean_acc:.2%}")


def test_heart_rate_mae():
    """Test heart rate MAE calculation."""
    # Perfect predictions
    predictions = np.array([72.0, 75.0, 68.0])
    references = np.array([72.0, 75.0, 68.0])
    
    mae = evaluate_heart_rate_accuracy(predictions, references)
    assert mae == 0.0
    
    # With errors
    predictions = np.array([70.0, 73.0, 70.0])
    mae = evaluate_heart_rate_accuracy(predictions, references)
    assert mae > 0.0


def test_group_consistency():
    """Test that groups are kept together in splits."""
    # Create data with known group structure
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    
    # Validate
    is_valid = validate_group_kfold(groups, n_splits=3)
    assert is_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
