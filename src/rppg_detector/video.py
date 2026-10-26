"""Video reading functionality for rPPG analysis."""

import cv2
import numpy as np
from pathlib import Path


def read_video_frames(video_path: str) -> tuple[np.ndarray, float]:
    """
    Read all frames from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frames, fps) where:
        - frames: numpy array of shape (num_frames, height, width, 3)
        - fps: frames per second as a float
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If the video cannot be read
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid frame rate detected: {fps}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")
    
    return np.array(frames), fps


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }
