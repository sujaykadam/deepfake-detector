"""Synthetic feature data for the tests."""

from typing import Tuple

import numpy as np


def create_synthetic_training_data(n_real: int = 100, n_fake: int = 100, 
                                   n_windows: int = 1, n_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for testing.
    
    Args:
        n_real: Number of real video samples
        n_fake: Number of fake video samples
        n_windows: Number of windows per video
        n_features: Number of features per window
        
    Returns:
        Tuple of (X, y) where:
        - X: Feature vectors of shape (n_real + n_fake, n_windows * n_features)
        - y: Labels array of shape (n_real + n_fake,)
    """
    # Real videos: higher coherence, more consistent signals
    real_features = np.random.randn(n_real, n_windows, n_features)
    real_features[:, :, 0] += 0.5  # Higher mean coherence
    real_features[:, :, 1] += 0.3  # Higher min coherence
    real_features[:, :, 2] -= 5.0  # Lower HR spread
    real_features[:, :, 3] += 0.2  # Better quality
    
    # Fake videos: lower coherence, less consistent signals
    fake_features = np.random.randn(n_fake, n_windows, n_features)
    fake_features[:, :, 0] -= 0.3  # Lower mean coherence
    fake_features[:, :, 1] -= 0.2  # Lower min coherence
    fake_features[:, :, 2] += 3.0  # Higher HR spread
    fake_features[:, :, 3] -= 0.1  # Lower quality
    
    # Reshape to (samples, features)
    X = np.vstack([real_features.reshape(n_real, -1), 
                   fake_features.reshape(n_fake, -1)])
    
    # Labels: 0 = real, 1 = fake
    y = np.array([0] * n_real + [1] * n_fake)
    
    return X, y
