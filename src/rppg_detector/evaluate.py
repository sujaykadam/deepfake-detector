"""Evaluation functions for heart rate estimation and deepfake detection."""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             precision_score, recall_score)
from typing import Tuple, List, Dict
from rppg_detector.classify import DeepfakeClassifier, train_classifier
from rppg_detector.coherence import extract_window_features
from rppg_detector.heartrate import estimate_heart_rate


def evaluate_heart_rate_accuracy(predictions: np.ndarray, 
                                 references: np.ndarray) -> float:
    """
    Evaluate heart rate estimation accuracy.
    
    Args:
        predictions: Predicted heart rates in BPM
        references: Reference heart rates in BPM
        
    Returns:
        Mean absolute error in BPM
    """
    return mean_absolute_error(references, predictions)


def _split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Return (accuracy, precision, recall) of 0/1 predictions, each in 0..1."""
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0))


def cross_validate_classifier(X: np.ndarray, y: np.ndarray, 
                             groups: np.ndarray, n_splits: int = 5) -> Dict:
    """
    Cross-validate classifier using GroupKFold, so no person_id is on both sides.
    
    Args:
        X: Feature vectors of shape (samples, features)
        y: Labels, 0 = real, 1 = fake
        groups: Person IDs of shape (samples,)
        n_splits: Number of cross-validation splits
        
    Returns:
        Dictionary of per-split lists (train_/test_ accuracy, precision, recall)
        plus a 'summary' dictionary of means and standard deviations
    """
    results = {f'{part}_{name}': [] for part in ('train', 'test')
               for name in ('accuracy', 'precision', 'recall')}
    
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        classifier = DeepfakeClassifier()
        classifier.train(X[train_idx], y[train_idx])
        for part, idx in (('train', train_idx), ('test', test_idx)):
            metrics = _split_metrics(y[idx], classifier.predict(X[idx]))
            for name, value in zip(('accuracy', 'precision', 'recall'), metrics):
                results[f'{part}_{name}'].append(value)
    
    results['summary'] = {
        'mean_train_accuracy': np.mean(results['train_accuracy']),
        'mean_test_accuracy': np.mean(results['test_accuracy']),
        'std_train_accuracy': np.std(results['train_accuracy']),
        'std_test_accuracy': np.std(results['test_accuracy']),
        'mean_test_precision': np.mean(results['test_precision']),
        'mean_test_recall': np.mean(results['test_recall']),
    }
    return results


def evaluate_with_group_kfold(X: np.ndarray, y: np.ndarray, 
                             groups: np.ndarray) -> Dict:
    """
    Evaluate classifier using GroupKFold cross-validation (5 splits).
    
    Args:
        X: Feature vectors of shape (samples, features)
        y: Labels, 0 = real, 1 = fake
        groups: Person IDs of shape (samples,)
        
    Returns:
        Same dictionary as cross_validate_classifier
    """
    return cross_validate_classifier(X, y, groups)


def validate_group_kfold(groups: np.ndarray, n_splits: int = 5) -> bool:
    """
    Validate that GroupKFold correctly keeps groups together.
    
    Args:
        groups: Group labels
        n_splits: Number of splits
        
    Returns:
        True if all groups are kept together in splits
    """
    gkf = GroupKFold(n_splits=n_splits)
    unique_groups = np.unique(groups)
    
    for train_idx, test_idx in gkf.split(groups, groups, groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        
        # Check that no group appears in both train and test
        overlap = train_groups & test_groups
        if len(overlap) > 0:
            return False
    
    return True


def evaluate_detector_on_video(signals: np.ndarray, fs: float, 
                              classifier: DeepfakeClassifier) -> Dict:
    """
    Evaluate detector on a single video.
    
    Args:
        signals: Video signals array of shape (frames, patches)
        fs: Sampling frequency in Hz
        classifier: Trained classifier
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract features
    features = extract_window_features(signals, fs)
    
    # Classify each window
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    
    # Calculate statistics
    fake_fraction = np.mean(predictions)
    avg_prob_fake = np.mean(probabilities[:, 1])
    
    return {
        'fake_fraction': fake_fraction,
        'avg_prob_fake': avg_prob_fake,
        'num_windows': len(predictions),
        'windows_voted_fake': int(np.sum(predictions)),
        'windows_voted_real': int(len(predictions) - np.sum(predictions))
    }
