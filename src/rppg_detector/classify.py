"""Classification model for deepfake detection."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from typing import Tuple, List
from rppg_detector.coherence import extract_window_features


class DeepfakeClassifier:
    """
    Logistic regression classifier for deepfake detection.
    
    Trains on window feature vectors to distinguish real from fake videos.
    """
    
    def __init__(self, window_seconds: float = 10.0, step_seconds: float = 5.0):
        """
        Initialize the classifier.
        
        Args:
            window_seconds: Sliding window length in seconds (default: 10)
            step_seconds: Sliding window step in seconds (default: 5)
        """
        # Features are on different scales (coherence 0-1, BPM spread 0-100), so scale
        # them, and weight classes so a larger real set does not swamp the fake class
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.is_trained = False
    
    def extract_features(self, signals: np.ndarray, fs: float) -> np.ndarray:
        """
        Extract features from signals for classification.
        
        Args:
            signals: Signals array of shape (frames, patches)
            fs: Sampling frequency in Hz
            
        Returns:
            Feature vectors array
        """
        return extract_window_features(signals, fs, self.window_seconds,
                                       self.step_seconds)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier on feature vectors.
        
        Args:
            X: Feature vectors array of shape (samples, features)
            y: Labels array of shape (samples,) with 0=real, 1=fake
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for feature vectors.
        
        Args:
            X: Feature vectors array of shape (samples, features)
            
        Returns:
            Predicted labels array of shape (samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for feature vectors.
        
        Args:
            X: Feature vectors array of shape (samples, features)
            
        Returns:
            Probability array of shape (samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def classify_video(self, signals: np.ndarray, fs: float) -> Tuple[str, float]:
        """
        Classify an entire video based on window voting.
        
        Args:
            signals: Signals array of shape (frames, patches)
            fs: Sampling frequency in Hz
            
        Returns:
            Tuple of (prediction, score) where:
            - prediction: "real" or "fake"
            - score: fraction of windows voting fake
        """
        features = self.extract_features(signals, fs)
        
        if len(features) == 0:
            return "real", 0.5  # Default to real if no features
        
        predictions = self.predict(features)
        
        # Majority voting
        fake_fraction = np.mean(predictions)
        prediction = "fake" if fake_fraction > 0.5 else "real"
        
        return prediction, fake_fraction
    
    def save(self, path: str):
        """
        Save the classifier model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump({'model': self.model, 'window_seconds': self.window_seconds,
                     'step_seconds': self.step_seconds}, path)
    
    def load(self, path: str):
        """
        Load a classifier model from disk.
        
        Args:
            path: Path to load the model from
        """
        saved = joblib.load(path)
        self.model = saved['model']
        self.window_seconds = saved['window_seconds']
        self.step_seconds = saved['step_seconds']
        self.is_trained = True


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[DeepfakeClassifier, float]:
    """
    Train and evaluate a classifier.
    
    Args:
        X_train: Training feature vectors
        y_train: Training labels
        X_test: Testing feature vectors
        y_test: Testing labels
        
    Returns:
        Tuple of (classifier, accuracy)
    """
    classifier = DeepfakeClassifier()
    classifier.train(X_train, y_train)
    
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return classifier, accuracy
