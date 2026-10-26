"""Command-line interface for rppg_detector."""

import argparse
import sys
import numpy as np

from rppg_detector.face import FaceDetector
from rppg_detector.signals import video_to_green_signals
from rppg_detector.heartrate import estimate_heart_rate
from rppg_detector.classify import DeepfakeClassifier
from rppg_detector.dataset import build_window_dataset, load_manifest_csv

FACE_MODEL_HELP = 'Path to face_landmarker.task (or set RPPG_FACE_MODEL)'


def cmd_hr(args):
    """Handle the 'hr' command."""
    signals, fps = video_to_green_signals(args.video, FaceDetector(args.face_model))
    print(f"Loaded video: {args.video}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {len(signals)}")
    
    hr, quality = estimate_heart_rate(signals, fps)
    print(f"Estimated heart rate: {hr:.1f} BPM")
    print(f"Quality score: {quality:.4f}")
    return 0


def cmd_train(args):
    """Handle the 'train' command."""
    manifest = load_manifest_csv(args.manifest)
    print(f"Loaded {len(manifest)} videos from {args.manifest}")
    
    X, y, _ = build_window_dataset(manifest, FaceDetector(args.face_model),
                                   args.window, args.step)
    print(f"Extracted {len(X)} windows ({int(np.sum(y))} fake, {int(np.sum(y == 0))} real)")
    
    classifier = DeepfakeClassifier(args.window, args.step)
    classifier.train(X, y)
    classifier.save(args.out)
    print(f"Model saved to {args.out}")
    return 0


def cmd_predict(args):
    """Handle the 'predict' command."""
    classifier = DeepfakeClassifier()
    classifier.load(args.model)
    print(f"Loaded model from {args.model}")
    
    signals, fps = video_to_green_signals(args.video, FaceDetector(args.face_model))
    prediction, score = classifier.classify_video(signals, fps)
    print(f"Prediction: {prediction}")
    print(f"Score (fraction of windows voting fake): {score:.4f}")
    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='rppg_detector: rPPG-based deepfake detector'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # HR command
    hr_parser = subparsers.add_parser('hr', help='Estimate heart rate from video')
    hr_parser.add_argument('video', help='Path to video file')
    hr_parser.add_argument('--face-model', help=FACE_MODEL_HELP)
    hr_parser.set_defaults(func=cmd_hr)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train classifier')
    train_parser.add_argument('manifest', help='Path to manifest CSV')
    train_parser.add_argument('--out', default='model.pkl', help='Output model path')
    train_parser.add_argument('--window', type=float, default=10.0,
                              help='Window length in seconds (default: 10)')
    train_parser.add_argument('--step', type=float, default=5.0,
                              help='Window step in seconds (default: 5)')
    train_parser.add_argument('--face-model', help=FACE_MODEL_HELP)
    train_parser.set_defaults(func=cmd_train)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Classify video')
    predict_parser.add_argument('video', help='Path to video file')
    predict_parser.add_argument('--model', default='model.pkl', help='Path to model file')
    predict_parser.add_argument('--face-model', help=FACE_MODEL_HELP)
    predict_parser.set_defaults(func=cmd_predict)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
