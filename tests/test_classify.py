"""Tests for classify.py module."""

import numpy as np
import pytest
import tempfile
import os
from rppg_detector.classify import DeepfakeClassifier, train_classifier
from tests.synthetic import create_synthetic_training_data


def test_classifier_initialization():
    """Test classifier initialization."""
    classifier = DeepfakeClassifier()
    assert not classifier.is_trained


def test_classifier_training():
    """Test classifier training on synthetic data."""
    # Create synthetic training data
    X, y = create_synthetic_training_data(n_real=100, n_fake=100)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train classifier
    classifier = DeepfakeClassifier()
    classifier.train(X_train, y_train)
    
    # Verify training worked
    assert classifier.is_trained
    
    # Make predictions
    predictions = classifier.predict(X_test)
    
    # Verify predictions
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
    
    # Check accuracy (should be > 90% for well-separated data)
    accuracy = np.mean(predictions == y_test)
    assert accuracy >= 0.90
    print(f"Accuracy: {accuracy:.2%}")


def test_classifier_save_load():
    """Test classifier save and load functionality."""
    # Create synthetic data
    X, y = create_synthetic_training_data(n_real=50, n_fake=50)
    
    # Train classifier
    classifier = DeepfakeClassifier()
    classifier.train(X, y)
    
    # Save classifier
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    classifier.save(temp_path)
    
    # Load classifier
    loaded_classifier = DeepfakeClassifier()
    loaded_classifier.load(temp_path)
    
    # Verify loaded classifier works
    predictions = loaded_classifier.predict(X)
    assert len(predictions) == len(y)
    
    # Cleanup
    os.unlink(temp_path)


def test_classify_video():
    """Test video classification."""
    # Create synthetic data
    X, y = create_synthetic_training_data(n_real=100, n_fake=100)
    
    # Train classifier
    classifier = DeepfakeClassifier()
    classifier.train(X, y)
    
    # Create test signals
    fs = 30.0
    signals = np.random.randn(900, 3)  # 30 seconds at 30 fps
    
    # Classify video
    prediction, score = classifier.classify_video(signals, fs)
    
    # Verify results
    assert prediction in ['real', 'fake']
    assert 0.0 <= score <= 1.0


def test_predict_proba():
    """Test probability prediction."""
    # Create synthetic data
    X, y = create_synthetic_training_data(n_real=50, n_fake=50)
    
    # Train classifier
    classifier = DeepfakeClassifier()
    classifier.train(X, y)
    
    # Get probabilities
    proba = classifier.predict_proba(X[:10])
    
    # Verify results
    assert proba.shape == (10, 2)  # 10 samples, 2 classes
    assert np.allclose(np.sum(proba, axis=1), 1.0)  # Probabilities sum to 1


def test_classifier_with_realistic_data():
    """Test classifier with more realistic synthetic data."""
    # Create data with more features
    n_samples = 200
    n_features = 20  # 5 windows * 4 features
    
    # Real videos: higher coherence values
    real_features = np.random.randn(n_samples // 2, n_features)
    real_features[:, :5] += 0.5  # Mean coherence
    real_features[:, 5:10] += 0.3  # Min coherence
    real_features[:, 10:15] -= 5.0  # Lower HR spread
    real_features[:, 15:] += 0.2  # Better quality
    
    # Fake videos: lower coherence values
    fake_features = np.random.randn(n_samples // 2, n_features)
    fake_features[:, :5] -= 0.3  # Lower mean coherence
    fake_features[:, 5:10] -= 0.2  # Lower min coherence
    fake_features[:, 10:15] += 3.0  # Higher HR spread
    fake_features[:, 15:] -= 0.1  # Lower quality
    
    X = np.vstack([real_features, fake_features])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Train and test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    classifier = DeepfakeClassifier()
    classifier.train(X_train, y_train)
    
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    # Should achieve > 90% accuracy
    assert accuracy >= 0.90
    print(f"Realistic data accuracy: {accuracy:.2%}")


def test_classifier_error_handling():
    """Test classifier error handling."""
    classifier = DeepfakeClassifier()
    
    # Test predict before training
    X = np.random.randn(10, 20)
    with pytest.raises(ValueError):
        classifier.predict(X)
    
    # Test predict_proba before training
    with pytest.raises(ValueError):
        classifier.predict_proba(X)


def test_train_classifier_function():
    """Test the train_classifier function."""
    # Create synthetic data
    X, y = create_synthetic_training_data(n_real=100, n_fake=100)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train using function
    classifier, accuracy = train_classifier(X_train, y_train, X_test, y_test)
    
    # Verify results
    assert classifier.is_trained
    assert accuracy >= 0.90
    print(f"Function train accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
