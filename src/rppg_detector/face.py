"""Face detection and skin patch extraction using MediaPipe Face Landmarker."""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Face Landmarker returns 478 landmarks (the Face Mesh topology).
# Each patch below is a ring of landmarks that outlines a patch of bare skin;
# the mean colour is taken over the filled outline only, not the bounding box.

# Forehead: 10 top-centre, 67/69/104/108/109 left brow ridge up, 151 mid-forehead,
# 337/338/297/299 right side.
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 337, 299, 297, 338]

# Left cheek (the subject's right cheek, on the viewer's left of the image).
LEFT_CHEEK_LANDMARKS = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]

# Right cheek (the subject's left cheek, on the viewer's right of the image).
RIGHT_CHEEK_LANDMARKS = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]

# Skin patches as a list of landmark index lists
SKIN_PATCHES = [FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]

# Environment variable that names the face_landmarker.task model file.
MODEL_ENV_VAR = "RPPG_FACE_MODEL"


class FaceDetector:
    """
    Face detector using MediaPipe Face Landmarker.

    Provides methods to detect faces and extract skin patches from video frames.
    The model file is loaded on the first detection, so creating a detector
    (or stubbing detect_faces in tests) does not need MediaPipe or the model.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the face detector.

        Args:
            model_path: Path to face_landmarker.task. If None, the path in the
                environment variable RPPG_FACE_MODEL is used.
        """
        self._model_path = model_path or os.environ.get(MODEL_ENV_VAR)
        self._landmarker = None
        self._previous_landmarks = None
        self._dropped_frames = 0

    def _load(self):
        """Create the MediaPipe landmarker, raising a clear error if no model."""
        if not self._model_path or not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found: {self._model_path!r}. "
                f"Pass model_path or set {MODEL_ENV_VAR} to face_landmarker.task."
            )
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_faces(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the face in a single frame.

        Args:
            frame: Input image of shape (height, width, 3) in BGR order (OpenCV)

        Returns:
            Array of shape (478, 2) with landmark (x, y) in pixels,
            or None if no face is found
        """
        import mediapipe as mp

        if self._landmarker is None:
            self._load()
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        points = result.face_landmarks[0]
        return np.array([[p.x * width, p.y * height] for p in points])

    def get_skin_patches(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute the mean colour of each skin patch in a frame.

        Args:
            frame: Input image of shape (height, width, 3), any channel order
            landmarks: Landmark (x, y) pixel coordinates, shape (478, 2)

        Returns:
            Array of shape (patches, 3) with the mean of each channel, in the
            same channel order as the frame, or None if landmarks is None
        """
        if landmarks is None:
            return None

        patches = []
        for patch_landmarks in SKIN_PATCHES:
            outline = landmarks[patch_landmarks].astype(np.int32)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(outline), 1)
            pixels = frame[mask.astype(bool)]
            patches.append(pixels.mean(axis=0) if len(pixels) else np.zeros(3))
        return np.array(patches)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Process a single frame and extract skin patches.

        Args:
            frame: Input image of shape (height, width, 3)

        Returns:
            Tuple of (skin_patches, is_dropped) where:
            - skin_patches: Array of shape (3, 3) with mean colour per patch,
              from the previous frame's landmarks if no face is found now
            - is_dropped: True if this frame had to use previous landmarks
        """
        landmarks = self.detect_faces(frame)

        if landmarks is not None:
            self._previous_landmarks = landmarks
            self._dropped_frames = 0
        else:
            self._dropped_frames += 1

        if self._previous_landmarks is not None:
            patches = self.get_skin_patches(frame, self._previous_landmarks)
            return patches, self._dropped_frames > 0
        return None, True

    def reset(self):
        """Reset the detector state."""
        self._previous_landmarks = None
        self._dropped_frames = 0


def get_skin_patch_indices() -> List[List[int]]:
    """
    Return the skin patch landmark indices.

    Returns:
        List of landmark index lists for each skin patch
    """
    return SKIN_PATCHES
