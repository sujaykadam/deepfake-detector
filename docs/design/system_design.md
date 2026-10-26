# rppg_detector System Design

## Implementation approach

We will implement a Python package for rPPG-based deepfake detection using classical signal processing and machine learning. The system follows a modular pipeline architecture where each module has a single responsibility:

1. **Video Processing**: Uses OpenCV to read frames and extract frame rate
2. **Face Detection**: Uses MediaPipe Face Landmarker to locate 478 facial landmarks
3. **Signal Extraction**: Computes RGB values from skin patches (forehead, cheeks)
4. **Heart Rate Estimation**: FFT-based analysis to find heartbeat frequency
5. **Coherence Analysis**: Measures agreement between signals from different skin regions
6. **Classification**: Logistic regression on window features to detect deepfakes
7. **Evaluation**: GroupKFold cross-validation with person_id grouping
8. **CLI**: Command-line interface with argparse for three commands

**Key Design Decisions:**
- CPU-only processing (no GPU/deep learning)
- Sliding window analysis (10s windows, 5s step)
- Green channel as primary signal source (most sensitive to blood flow)
- FFT-based heart rate estimation for simplicity and speed
- Person-based grouping in cross-validation to prevent data leakage

## File list

- rppg_detector/__init__.py
- rppg_detector/video.py
- rppg_detector/face.py
- rppg_detector/signals.py
- rppg_detector/heartrate.py
- rppg_detector/coherence.py
- rppg_detector/classify.py
- rppg_detector/evaluate.py
- rppg_detector/cli.py
- tests/__init__.py
- tests/test_video.py
- tests/test_face.py
- tests/test_signals.py
- tests/test_heartrate.py
- tests/test_coherence.py
- tests/test_classify.py
- tests/test_evaluate.py
- pyproject.toml
- README.md
- check.sh

## Data structures and interfaces

classDiagram
    class Main {
        <<entry point>>
        +main() int
    }
    
    class VideoReader {
        -video_path: str
        +read_video_frames() tuple[np.ndarray, float]
        +get_video_info() dict
    }
    
    class FaceDetector {
        -_previous_landmarks: Optional[np.ndarray]
        -_dropped_frames: int
        +detect_faces() Optional[np.ndarray]
        +get_skin_patches() np.ndarray
        +process_frame() Tuple[Optional[np.ndarray], bool]
        +reset()
    }
    
    class SignalExtractor {
        +extract_rgb_signals() np.ndarray
        +get_green_channel() np.ndarray
        +compute_mean_rgb_per_patch() np.ndarray
        +extend_signal_methods() np.ndarray
        +normalize_signals() np.ndarray
    }
    
    class HeartRateEstimator {
        +detrend() np.ndarray
        +bandpass_filter() np.ndarray
        +estimate_heart_rate() Tuple[float, float]
        +estimate_heart_rate_per_patch() np.ndarray
    }
    
    class CoherenceAnalyzer {
        +compute_coherence_matrix() np.ndarray
        +extract_window_features() np.ndarray
        +get_window_boundaries() List[Tuple[int, int]]
    }
    
    class DeepfakeClassifier {
        -model: LogisticRegression
        +extract_features() np.ndarray
        +train()
        +predict() np.ndarray
        +predict_proba() np.ndarray
        +classify_video() Tuple[str, float]
        +save()
        +load()
    }
    
    class Evaluator {
        +evaluate_heart_rate_accuracy() float
        +cross_validate_classifier() Dict[str, List[float]]
        +evaluate_with_group_kfold() Dict
        +validate_group_kfold() bool
        +evaluate_detector_on_video() Dict
        +load_manifest_csv() pd.DataFrame
        +prepare_training_data() Tuple[np.ndarray, np.ndarray, np.ndarray]
    }
    
    class CLI {
        +main() int
        +cmd_hr()
        +cmd_train()
        +cmd_predict()
    }
    
    class Constants {
        <<constant>>
        +FOREHEAD_LANDMARKS: List[int]
        +LEFT_CHEEK_LANDMARKS: List[int]
        +RIGHT_CHEEK_LANDMARKS: List[int]
        +SKIN_PATCHES: List[List[int]]
    }

    Main --> CLI
    CLI --> VideoReader
    CLI --> FaceDetector
    CLI --> SignalExtractor
    CLI --> HeartRateEstimator
    CLI --> CoherenceAnalyzer
    CLI --> DeepfakeClassifier
    CLI --> Evaluator
    
    FaceDetector --> Constants
    SignalExtractor --> FaceDetector
    SignalExtractor --> Constants
    HeartRateEstimator --> SignalExtractor
    CoherenceAnalyzer --> HeartRateEstimator
    DeepfakeClassifier --> CoherenceAnalyzer
    Evaluator --> DeepfakeClassifier
    Evaluator --> HeartRateEstimator
    
    note right of Constants
        Landmark indices for skin patches
        based on MediaPipe Face Mesh (478 points)
    end note

## Program call flow

### Heart Rate Estimation Pipeline (rppg_detector hr command)

sequenceDiagram
    participant CLI as CLI (cmd_hr)
    participant VR as VideoReader
    participant FD as FaceDetector
    participant SE as SignalExtractor
    participant HR as HeartRateEstimator
    
    CLI->>VR: read_video_frames(video_path)
    VR-->>CLI: frames (np.ndarray), fps (float)
    
    CLI->>FD: FaceDetector()
    CLI->>SE: extract_rgb_signals(frames, FD)
    
    loop For each frame
        FD->>FD: detect_faces(frame)
        FD-->>SE: landmarks (Optional[np.ndarray])
        SE->>SE: get_skin_patches(frame, landmarks)
        SE-->>FD: skin_patches (Optional[np.ndarray])
    end
    
    SE-->>CLI: signals (frames, patches, 3)
    CLI->>SE: get_green_channel(signals)
    SE-->>CLI: green_signals (frames, patches)
    
    CLI->>HR: estimate_heart_rate(green_signals, fps)
    HR->>HR: detrend(signals)
    HR->>HR: bandpass_filter(detrended, fs=fps)
    HR->>HR: np.fft.rfft(filtered)
    HR->>HR: find_peak_frequency()
    HR-->>CLI: (heart_rate_bpm, quality_score)
    
    CLI->>CLI: print results

### Classification Pipeline (rppg_detector predict command)

sequenceDiagram
    participant CLI as CLI (cmd_predict)
    participant VR as VideoReader
    participant FD as FaceDetector
    participant SE as SignalExtractor
    participant CA as CoherenceAnalyzer
    participant DC as DeepfakeClassifier
    
    CLI->>DC: DeepfakeClassifier().load(model_path)
    CLI->>VR: read_video_frames(video_path)
    VR-->>CLI: frames (np.ndarray), fps (float)
    
    CLI->>FD: FaceDetector()
    CLI->>SE: extract_rgb_signals(frames, FD)
    
    loop For each frame
        FD->>FD: detect_faces(frame)
        FD-->>SE: landmarks
        SE->>SE: get_skin_patches(frame, landmarks)
    end
    
    SE-->>CLI: signals (frames, patches, 3)
    CLI->>CA: extract_window_features(signals, fps)
    
    loop For each window
        CA->>CA: compute_coherence_matrix(window)
        CA->>CA: extract_features(window)
    end
    
    CA-->>CLI: features (num_windows, 4)
    CLI->>DC: classify_video(signals, fps)
    
    DC->>DC: extract_features(signals, fps)
    DC->>DC: predict(features)
    DC->>DC: majority voting
    DC-->>CLI: (prediction, fake_fraction)
    
    CLI->>CLI: print prediction and score

### Training Pipeline (rppg_detector train command)

sequenceDiagram
    participant CLI as CLI (cmd_train)
    participant EV as Evaluator
    participant DC as DeepfakeClassifier
    
    CLI->>EV: load_manifest_csv(manifest_path)
    EV-->>CLI: manifest_df (DataFrame)
    
    CLI->>EV: prepare_training_data(manifest_df)
    EV->>EV: read paths from CSV
    EV-->>CLI: (X, y, groups)
    
    CLI->>DC: classifier = DeepfakeClassifier()
    CLI->>DC: classifier.train(X, y)
    
    DC->>DC: model.fit(X, y)
    DC-->>CLI: trained model
    
    CLI->>DC: classifier.save(out_path)
    DC->>DC: joblib.dump(model, path)
    DC-->>CLI: model saved
    
    CLI->>CLI: print success message

### Cross-Validation Pipeline (evaluate.py functions)

sequenceDiagram
    participant EV as Evaluator
    participant DC as DeepfakeClassifier
    participant GKF as GroupKFold
    
    EV->>EV: cross_validate_classifier(X, y, groups)
    EV->>GKF: GroupKFold.split(X, y, groups)
    
    loop For each fold
        GKF-->>EV: train_idx, test_idx
        EV->>EV: X_train, X_test = X[train_idx], X[test_idx]
        EV->>EV: y_train, y_test = y[train_idx], y[test_idx]
        
        EV->>DC: classifier.train(X_train, y_train)
        EV->>DC: predictions = classifier.predict(X_test)
        EV->>EV: calculate_metrics(predictions, y_test)
    end
    
    EV->>EV: calculate_summary_statistics()
    EV-->>EV: results dictionary

## Error Handling

### Video Reading Errors
- `FileNotFoundError`: Raised when video file doesn't exist
- `RuntimeError`: Raised when video cannot be opened or has invalid frame rate
- `RuntimeError`: Raised when no frames are read from video

### Face Detection Errors
- Returns `None` landmarks when no face detected
- Carries forward previous landmarks (with dropped frame count)
- Graceful degradation with quality metrics

### Signal Processing Errors
- Handles insufficient data gracefully
- Returns 0 BPM and 0 quality when no valid frequency found
- Normalizes signals to avoid division by zero

### Classification Errors
- Raises `ValueError` if model not trained before prediction
- Handles empty feature vectors by returning default "real" prediction

## Edge Cases

1. **Short videos** (< 10 seconds): Returns coarse heart rate estimates
2. **No face detected**: Carries forward previous landmarks, tracks dropped frames
3. **Very low quality signals**: Returns low quality scores
4. **Multiple people in frame**: Uses first detected face only
5. **Frame rate variations**: Uses actual FPS from video metadata
6. **Empty windows**: Skips windows with less than 50% of expected frames

## CLI Interface Design

```bash
# Heart rate estimation
rppg_detector hr VIDEO

# Model training
rppg_detector train MANIFEST_CSV --out MODEL

# Video classification
rppg_detector predict VIDEO --model MODEL

# Help
rppg_detector --help
rppg_detector hr --help
rppg_detector train --help
rppg_detector predict --help
```

## Testing Strategy

1. **Unit Tests**: Each module tested independently
2. **Integration Tests**: Full pipeline tested with synthetic data
3. **Signal Validation**: 72 BPM signal must be detected within 3 BPM
4. **Coherence Validation**: Shared-pulse signals must score higher than unrelated
5. **GroupKFold Validation**: Person IDs never split across train/test
6. **CLI Validation**: All commands work with synthetic data

## Performance Considerations

- Processes frames sequentially (memory efficient)
- Uses NumPy vectorized operations where possible
- FFT-based heart rate estimation (O(n log n))
- No GPU requirements (CPU-only)
- Memory usage: O(frames × height × width × 3)

## Extensibility Points

1. **Signal Extraction**: `extend_signal_methods()` allows adding CHROM, POS methods
2. **Feature Engineering**: `extract_window_features()` can be extended with more features
3. **Classification**: `DeepfakeClassifier` can be swapped for other models
4. **Face Detection**: `FaceDetector` interface allows different detector implementations

## Known Limitations (per requirements)

1. Compressed or moving video degrades pulse signal quality
2. Clips shorter than 10 seconds give coarse heart rate estimates
3. Skin tone affects heart rate reader performance
4. Advanced deepfakes can copy pulse signals from source videos
5. Homemade detector will score well below FakeCatcher's 96% accuracy
