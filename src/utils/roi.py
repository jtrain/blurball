"""
Region of Interest (ROI) cropping for portrait video inference.

Allows extracting a portrait-oriented crop from videos to focus on relevant areas
(e.g., stumps region in cricket) and rescaling to model input dimensions.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from utils.image import get_affine_transform


@dataclass
class ROIConfig:
    """Configuration for ROI extraction."""

    # Crop dimensions (width, height in pixels)
    width: int
    height: int

    # Center position as fraction of video frame (0.0 to 1.0)
    center_x: float = 0.5  # Horizontal center
    center_y: float = 0.5  # Vertical center

    # Model input dimensions (what to rescale to)
    model_inp_width: int = 512
    model_inp_height: int = 288


class ROICropper:
    """Handles ROI cropping and resizing."""

    @staticmethod
    def get_crop_box(
        roi: ROIConfig, video_width: int, video_height: int
    ) -> tuple[int, int, int, int]:
        """
        Get crop box coordinates (x1, y1, x2, y2) in pixel space.
        """
        x1 = int(roi.center_x * video_width - roi.width / 2)
        y1 = int(roi.center_y * video_height - roi.height / 2)
        x2 = x1 + roi.width
        y2 = y1 + roi.height

        # Clamp to frame boundaries
        x1 = max(0, min(x1, video_width - 1))
        y1 = max(0, min(y1, video_height - 1))
        x2 = max(x1 + 1, min(x2, video_width))
        y2 = max(y1 + 1, min(y2, video_height))

        return x1, y1, x2, y2

    @staticmethod
    def crop_frame(frame: np.ndarray, roi: ROIConfig) -> np.ndarray:
        """Crop frame to ROI region."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = ROICropper.get_crop_box(roi, w, h)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def crop_and_resize(frame: np.ndarray, roi: ROIConfig) -> np.ndarray:
        """Crop frame to ROI and resize to model input dimensions."""
        cropped = ROICropper.crop_frame(frame, roi)
        # Resize to model dimensions
        resized = cv2.resize(
            cropped,
            (roi.model_inp_width, roi.model_inp_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized


class ROITransform:
    """Compute affine transformations for ROI crops."""

    @staticmethod
    def get_roi_affine_matrices(
        roi: "ROIConfig", video_width: int, video_height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward and inverse affine transforms for ROI.

        Returns:
            (trans_forward, trans_inverse) - both are 2x3 matrices
            forward: global -> model input space
            inverse: model input space -> global
        """
        # Get the crop box in global coordinates
        x1, y1, x2, y2 = ROICropper.get_crop_box(roi, video_width, video_height)
        roi_width_actual = x2 - x1
        roi_height_actual = y2 - y1

        # Forward transform: global -> ROI crop -> model input
        # This is a composition of: translate to crop origin, scale to model size
        trans_forward = np.array([
            [roi.model_inp_width / roi_width_actual, 0, -x1 * roi.model_inp_width / roi_width_actual],
            [0, roi.model_inp_height / roi_height_actual, -y1 * roi.model_inp_height / roi_height_actual]
        ], dtype=np.float32)

        # Inverse transform: model input -> ROI crop -> global
        # Scale model input back to ROI size, then translate to global position
        trans_inverse = np.array([
            [roi_width_actual / roi.model_inp_width, 0, x1],
            [0, roi_height_actual / roi.model_inp_height, y1]
        ], dtype=np.float32)

        return trans_forward, trans_inverse

    @staticmethod
    def create_transform_matrices_for_inference(
        roi: Optional["ROIConfig"],
        video_width: int,
        video_height: int,
        model_inp_width: int,
        model_inp_height: int,
        batch_size: int = 1,
        num_scales: int = 3,
    ) -> np.ndarray:
        """
        Create transform matrices for inference (inverse transforms).

        This unifies ROI and non-ROI preprocessing by always computing proper
        affine matrices that transform detections back to global coordinates.

        Args:
            roi: ROI configuration, or None for full-frame affine
            video_width: Original video width
            video_height: Original video height
            model_inp_width: Model input width
            model_inp_height: Model input height
            batch_size: Batch size
            num_scales: Number of scales in multi-scale output

        Returns:
            Transform matrices shaped (batch_size, num_scales, 2, 3)
            for applying in postprocessor
        """
        if roi is not None:
            # ROI case: use ROI transforms
            _, trans_inverse = ROITransform.get_roi_affine_matrices(
                roi, video_width, video_height
            )
            # Replicate across batch and scales
            trans = np.stack([trans_inverse for _ in range(num_scales)], axis=0)
            trans = np.stack([trans for _ in range(batch_size)], axis=0)
        else:
            # Non-ROI case: use center-based affine like before
            c = np.array([video_width / 2.0, video_height / 2.0], dtype=np.float32)
            s = max(video_height, video_width) * 1.0
            trans = np.stack([
                get_affine_transform(c, s, 0, [model_inp_width, model_inp_height], inv=1)
                for _ in range(num_scales)
            ], axis=0)
            trans = np.stack([trans for _ in range(batch_size)], axis=0)

        return trans


class ROIValidator:
    """Validates ROI configuration against video dimensions."""

    @staticmethod
    def validate(
        roi: ROIConfig, video_width: int, video_height: int
    ) -> tuple[bool, str]:
        """
        Validate ROI config against video dimensions.

        Returns (is_valid, message)
        """
        # Check ROI fits in frame
        roi_left = int(roi.center_x * video_width - roi.width / 2)
        roi_right = roi_left + roi.width
        roi_top = int(roi.center_y * video_height - roi.height / 2)
        roi_bottom = roi_top + roi.height

        if roi_left < 0 or roi_right > video_width:
            return (
                False,
                f"ROI width {roi.width} doesn't fit horizontally in {video_width}px frame",
            )

        if roi_top < 0 or roi_bottom > video_height:
            return (
                False,
                f"ROI height {roi.height} doesn't fit vertically in {video_height}px frame",
            )

        # Check aspect ratio compatibility
        roi_aspect = roi.width / roi.height
        model_aspect = roi.model_inp_width / roi.model_inp_height
        aspect_ratio = roi_aspect / model_aspect

        if aspect_ratio < 0.9 or aspect_ratio > 1.1:
            recommended_width = int(roi.height * model_aspect)
            recommended_height = int(roi.width / model_aspect)
            return False, (
                f"ROI aspect ratio {roi_aspect:.2f} differs from model {model_aspect:.2f} by {abs(aspect_ratio - 1) * 100:.1f}%. "
                f"Consider width={recommended_width} (height={roi.height}) or "
                f"height={recommended_height} (width={roi.width})"
            )

        return True, "ROI config is valid"

    @staticmethod
    def suggest_roi_for_portrait(
        video_width: int,
        video_height: int,
        model_inp_width: int = 512,
        model_inp_height: int = 288,
        use_full_width: bool = True,
    ) -> ROIConfig:
        """
        Suggest a sensible ROI for portrait video.

        Args:
            video_width: Input video width (typically smaller for portrait)
            video_height: Input video height (typically larger for portrait)
            model_inp_width: Model input width (default 512)
            model_inp_height: Model input height (default 288)
            use_full_width: If True, use full video width and scale height.
                          If False, use full height and scale width.

        Returns:
            ROIConfig with suggested dimensions
        """
        model_aspect = model_inp_width / model_inp_height

        if use_full_width:
            # Use full video width, scale height to match model aspect
            roi_width = video_width
            roi_height = min(int(video_width / model_aspect), video_height)
        else:
            # Use full video height, scale width to match model aspect
            roi_height = video_height
            roi_width = min(int(video_height * model_aspect), video_width)

        # Center the ROI vertically
        center_y = (
            0.5 if roi_height == video_height else 0.4
        )  # Bias slightly up for action

        return ROIConfig(
            width=roi_width,
            height=roi_height,
            center_x=0.5,
            center_y=center_y,
            model_inp_width=model_inp_width,
            model_inp_height=model_inp_height,
        )


class ROIVisualizer:
    """Visualize ROI on frames."""

    @staticmethod
    def draw_roi_preview(
        frame: np.ndarray, roi: ROIConfig, thickness: int = 2
    ) -> np.ndarray:
        """
        Draw ROI rectangle on frame.

        Args:
            frame: Input frame
            roi: ROI configuration
            thickness: Line thickness

        Returns:
            Frame with ROI rectangle drawn
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = ROICropper.get_crop_box(roi, w, h)

        # Draw green rectangle for ROI
        frame_vis = frame.copy()
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # Add text showing dimensions
        text = f"ROI {roi.width}x{roi.height} -> {roi.model_inp_width}x{roi.model_inp_height}"
        cv2.putText(
            frame_vis,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        return frame_vis

    @staticmethod
    def create_preview_grid(
        frame: np.ndarray,
        roi: ROIConfig,
        n_cols: int = 3,
    ) -> np.ndarray:
        """
        Create a preview grid showing original frame, ROI preview, and cropped result.

        Args:
            frame: Input frame
            roi: ROI configuration
            n_cols: Number of columns in grid

        Returns:
            Grid image
        """
        # Original frame with ROI drawn
        frame_with_roi = ROIVisualizer.draw_roi_preview(frame, roi)
        h_frame, w_frame = frame.shape[:2]
        frame_with_roi = cv2.resize(frame_with_roi, (512, 288))

        # Cropped frame
        cropped = ROICropper.crop_and_resize(frame, roi)

        # Side-by-side comparison
        comparison = np.hstack([frame_with_roi, cropped])

        return comparison

    @staticmethod
    def create_preview_grid_with_info(frame: np.ndarray, roi: ROIConfig) -> np.ndarray:
        """
        Create preview grid with center_y info overlay.

        Args:
            frame: Input frame
            roi: ROI configuration

        Returns:
            Grid image with info text
        """
        grid = ROIVisualizer.create_preview_grid(frame, roi)

        # Add info text overlay at bottom
        info_text = f"center_y: {roi.center_y:.3f}"
        h, w = grid.shape[:2]
        cv2.putText(
            grid,
            info_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return grid


def preview_roi_on_video(
    video_path: str,
    roi: ROIConfig | dict | None = None,
    n_frames: int = 10,
    display: bool = True,
) -> ROIConfig:
    """
    Preview ROI on sample frames from video.

    Args:
        video_path: Path to input video
        roi: ROI configuration (ROIConfig or dict with partial spec).
             If None, suggests one based on video dimensions.
             Dict can have partial fields: width, height, center_x, center_y,
             model_inp_width, model_inp_height. Missing dimensions will be auto-suggested.
        n_frames: Number of frames to sample
        display: If True, display frames using cv2.imshow (interactive)

    Returns:
        ROIConfig used (either provided or suggested)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle partial ROI specification (dict)
    if isinstance(roi, dict):
        partial_roi = roi
        model_width = partial_roi.get("model_inp_width", 512)
        model_height = partial_roi.get("model_inp_height", 288)

        # Suggest full ROI first
        suggested = ROIValidator.suggest_roi_for_portrait(w, h, model_width, model_height)

        # Override with user-specified values
        roi_width = partial_roi.get("width") or suggested.width
        roi_height = partial_roi.get("height") or suggested.height

        roi = ROIConfig(
            width=roi_width,
            height=roi_height,
            center_x=partial_roi.get("center_x", suggested.center_x),
            center_y=partial_roi.get("center_y", suggested.center_y),
            model_inp_width=model_width,
            model_inp_height=model_height,
        )
    # Suggest ROI if not provided
    elif roi is None:
        roi = ROIValidator.suggest_roi_for_portrait(w, h)

    # Validate ROI
    is_valid, msg = ROIValidator.validate(roi, w, h)
    print(f"ROI Validation: {msg}")
    if not is_valid:
        print("Error: ROI configuration does not fit in video frame.")
        print(f"Video is {w}x{h}, but ROI is {roi.width}x{roi.height}")
        raise ValueError(msg)

    print(f"Video dimensions: {w}x{h}")
    print(
        f"ROI config: width={roi.width}, height={roi.height}, center_x={roi.center_x}, center_y={roi.center_y}"
    )

    # Sample frames
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    current_roi = roi  # Working copy that can be modified interactively
    quit_preview = False

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if display:
            adjusting = True
            while adjusting:
                # Create preview with current ROI and center_y info
                preview = ROIVisualizer.create_preview_grid_with_info(
                    frame, current_roi
                )

                window_name = (
                    f"Frame {i + 1}/{n_frames} | y={current_roi.center_y:.3f} | "
                    f"UP/DOWN (±0.05), LEFT/RIGHT (±0.02), SPACE next, Q quit"
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(0)

                # Handle key presses
                if key == ord("q") or key == 27:  # q or ESC
                    quit_preview = True
                    adjusting = False
                elif key == ord(" ") or key == 13:  # SPACE or ENTER
                    adjusting = False
                elif key == ord("k"):  # UP arrow
                    current_roi.center_y = min(1.0, current_roi.center_y + 0.05)
                elif key == ord("j"):  # DOWN arrow
                    current_roi.center_y = max(0.0, current_roi.center_y - 0.05)
                elif key == ord("h"):  # LEFT arrow
                    current_roi.center_y = max(0.0, current_roi.center_y - 0.02)
                elif key == ord("l"):  # RIGHT arrow
                    current_roi.center_y = min(1.0, current_roi.center_y + 0.02)
                else:
                    adjusting = False
        else:
            print(f"Frame {i + 1}/{n_frames}: {frame_idx}")

        if quit_preview:
            break

    if display:
        cv2.destroyAllWindows()

    cap.release()
    return current_roi


# Helper function for use in inference config
def create_roi_from_dict(roi_dict: dict) -> ROIConfig:
    """Create ROIConfig from dictionary (for hydra config support)."""
    return ROIConfig(
        width=roi_dict["width"],
        height=roi_dict["height"],
        center_x=roi_dict.get("center_x", 0.5),
        center_y=roi_dict.get("center_y", 0.5),
        model_inp_width=roi_dict.get("model_inp_width", 512),
        model_inp_height=roi_dict.get("model_inp_height", 288),
    )
