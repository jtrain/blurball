import torch
from torch import nn
import logging
from .heatmap import HeatmapLoss

log = logging.getLogger(__name__)


class WeightedHeatmapLoss(nn.Module):
    """Applies frame-recency and spatial weighting to bounce event detection.

    Frames at the end of a clip (temporal recency) and detections lower on screen
    (higher y-coordinate) receive higher weight. Designed for cricket bounce events
    where the critical moment happens at the end of the flight with the ball low.

    Weighting is only applied when enabled in config and frame metadata is provided.
    Other datasets (volleyball, tennis, etc.) are unaffected.
    """

    def __init__(self, cfg):
        super().__init__()
        self._base_loss = HeatmapLoss(cfg)

        self._enable_bounce_weight = cfg.get("loss", {}).get("bounce_weight", {}).get("enabled", False)
        self._temporal_weight = cfg.get("loss", {}).get("bounce_weight", {}).get("temporal_weight", 1.0)
        self._spatial_weight = cfg.get("loss", {}).get("bounce_weight", {}).get("spatial_weight", 1.0)

        if self._enable_bounce_weight:
            log.info(f"WeightedHeatmapLoss enabled: temporal_weight={self._temporal_weight}, spatial_weight={self._spatial_weight}")

    def forward(self, inputs, targets, frame_metadata=None):
        """
        Args:
            inputs: model predictions (passed to base loss)
            targets: ground truth heatmaps, dict[scale] -> tensor
                    Each heatmap has shape [batch, num_frames, h, w]
            frame_metadata: dict with keys:
                - 'dataset_type': str, 'cricket' to apply weighting
                - 'num_frames': torch tensor of shape [batch_size], frames per sample
                - 'xy_gt': torch tensor of shape [batch_size, num_frames, 2], ball positions (optional)

        Returns:
            weighted loss scalar
        """
        # Check if weighting should be applied
        if not self._enable_bounce_weight or frame_metadata is None:
            return self._base_loss(inputs, targets)

        dataset_type = frame_metadata.get("dataset_type", "").lower()
        if dataset_type != "cricket":
            return self._base_loss(inputs, targets)

        num_frames_batch = frame_metadata.get("num_frames")
        xy_gt = frame_metadata.get("xy_gt")

        if num_frames_batch is None:
            return self._base_loss(inputs, targets)

        # Compute weights per frame: temporal (recency) boost
        try:
            # num_frames_batch is shape [batch_size] with scalar frame counts
            # In practice, all samples in a batch usually have the same num_frames
            num_frames = num_frames_batch[0].item() if hasattr(num_frames_batch[0], 'item') else int(num_frames_batch[0])

            # Create frame weights: [num_frames]
            # Weight increases from 0 to 1 across the clip (later frames get higher weight)
            frame_indices = torch.arange(num_frames, dtype=torch.float32, device=num_frames_batch.device)
            temporal_weight = (frame_indices + 1) / num_frames  # [1/N, 2/N, ..., 1]
            temporal_weight = temporal_weight.clamp(0, 1)

            # Add spatial weighting if available
            if xy_gt is not None and self._spatial_weight > 0:
                try:
                    # xy_gt shape: [batch_size, num_frames, 2]
                    # Coordinates are already normalized to [0, 1] by the dataloader
                    # Get y-coordinates (second element)
                    y_coords = xy_gt[..., 1]  # shape [batch_size, num_frames]
                    y_mean = y_coords.mean(dim=0)  # shape [num_frames], values in [0, 1]
                    # Clamp is still safe as double-check, but shouldn't be needed
                    spatial_weight = y_mean.clamp(0, 1)
                except Exception as e:
                    log.debug(f"Could not compute spatial weight: {e}")
                    spatial_weight = None
            else:
                spatial_weight = None

            # Combine temporal and spatial weights
            combined_weight = temporal_weight
            if spatial_weight is not None:
                # spatial_weight boosts importance for frames with ball lower on screen
                combined_weight = temporal_weight * (1 + self._spatial_weight * spatial_weight)

            # Normalize to mean=1 (so it modulates loss, not scales it)
            combined_weight = combined_weight / (combined_weight.mean().clamp(min=1e-8))

            # Apply weights to targets
            # targets is dict[scale] -> tensor of shape [batch, num_frames, h, w]
            weighted_targets = {}
            for scale_key, target_hm in targets.items():
                # Reshape weight for broadcasting: [num_frames] -> [1, num_frames, 1, 1]
                weight_expanded = combined_weight.view(1, num_frames, 1, 1)
                weighted_targets[scale_key] = target_hm * weight_expanded

            return self._base_loss(inputs, weighted_targets)

        except Exception as e:
            log.warning(f"Error applying bounce weighting: {e}, falling back to unweighted loss")
            return self._base_loss(inputs, targets)
