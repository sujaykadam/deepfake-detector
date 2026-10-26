import cv2
"""Tests for video.py module."""

import numpy as np
import pytest
from pathlib import Path
from rppg_detector.video import read_video_frames, get_video_info


def create_test_video(filepath: str, fps: float = 30.0, 
                     duration: float = 30.0) -> Path:
    """
    Create a test video file with known properties.
    
    Args:
        filepath: Path to save the video
        fps: Frames per second
        duration: Duration in seconds
        
    Returns:
        Path to the created video file
    """
    import cv2
    
    frame_count = int(fps * duration)
    height, width = 240, 320
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    # Create frames with skin-tone color
    for i in range(frame_count):
        # Create skin-colored frame with slight modulation
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [50, 100, 150]  # Skin tone color (BGR)
        out.write(frame)
    
    out.release()
    
    return Path(filepath)


def test_read_video_frames():
    """Test reading video frames."""
    # Create test video
    video_path = "/tmp/test_video.mp4"
    create_test_video(video_path, fps=30.0, duration=1.0)
    
    # Read frames
    frames, fps = read_video_frames(video_path)
    
    # Verify results
    assert frames.shape[0] == 30  # 1 second at 30 fps
    assert frames.shape[1] == 240  # Height
    assert frames.shape[2] == 320  # Width
    assert frames.shape[3] == 3   # RGB channels
    assert abs(fps - 30.0) < 0.1  # FPS should be close to 30
    
    # Cleanup
    Path(video_path).unlink()


def test_get_video_info():
    """Test getting video information."""
    # Create test video
    video_path = "/tmp/test_video_info.mp4"
    create_test_video(video_path, fps=25.0, duration=2.0)
    
    # Get info
    info = get_video_info(video_path)
    
    # Verify results
    assert abs(info['fps'] - 25.0) < 0.1
    assert info['frame_count'] == 50  # 2 seconds at 25 fps
    assert info['width'] == 320
    assert info['height'] == 240
    
    # Cleanup
    Path(video_path).unlink()


def test_nonexistent_file():
    """Test error handling for non-existent files."""
    with pytest.raises(FileNotFoundError):
        read_video_frames("/nonexistent/video.mp4")


def test_invalid_video_file():
    """Test error handling for invalid video files."""
    # Create a text file instead of video
    video_path = "/tmp/invalid_video.mp4"
    with open(video_path, 'w') as f:
        f.write("This is not a video file")
    
    with pytest.raises(RuntimeError):
        read_video_frames(video_path)
    
    # Cleanup
    Path(video_path).unlink()


def test_synthetic_72_bpm_video():
    """Test creating a video with 72 BPM modulation."""
    video_path = "/tmp/test_72bpm.mp4"
    fps = 30.0
    duration = 30.0  # 72 BPM = 1.2 Hz
    
    # Create video with green channel modulation at 1.2 Hz
    frame_count = int(fps * duration)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (320, 240))
    for i in range(frame_count):
        # Create frame with green channel modulated at 1.2 Hz
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        green_val = int(100 + 50 * np.sin(2 * np.pi * 1.2 * i / fps))
        frame[:] = [50, green_val, 150]  # Skin tone with modulated green
        out.write(frame)
    
    out.release()
    
    # Read and verify
    frames, read_fps = read_video_frames(video_path)
    assert abs(read_fps - fps) < 0.1
    assert len(frames) == frame_count
    
    # Cleanup
    Path(video_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
