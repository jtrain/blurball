# Bounce Event Weighting System

## Overview

The **Weighted Heatmap Loss** (`WeightedHeatmapLoss`) optimizes cricket bounce event detection by assigning higher importance to frames at the end of a clip and detections lower on the screen (where the ball is near the ground).

## Why This Matters

In cricket bounce detection, the early frames of a delivery are easy to detect—the ball is high and clear. The hard problem is pre/post bounce: distinguishing the exact moment and position when the ball hits the ground. This system trains the model to focus on that critical moment.

## How It Works

### 1. Temporal Weighting (Frame Recency)
- Frames at the END of a clip (later indices) receive HIGHER weight
- Weight increases linearly: frame 0 gets weight 1/N, frame N-1 gets weight 1.0
- This prioritizes learning the bounce moment, which is typically near the end

### 2. Spatial Weighting (Ball Position)  
- Detections with the ball LOWER on screen (higher y-coordinate) receive bonus weight
- Assumes y-coordinates are normalized to [0, 1] where 1 = bottom of screen
- Multiplier is applied on top of temporal weight

### 3. Per-Dataset Scoping
- Weighting is **only applied to cricket** (configurable)
- Volleyball, tennis, and other sports use standard unweighted loss
- Safe to enable without affecting other detectors

## Configuration

### Enable in `train_cricket.yaml`

```yaml
loss:
  name: weighted_heatmap           # Use weighted loss instead of plain heatmap
  sub_name: mse                     # Base loss type (mse, bce, combo, etc.)
  bounce_weight:
    enabled: true                   # Turn on/off weighting
    temporal_weight: 1.0            # How much to boost late frames (1.0 = moderate)
    spatial_weight: 0.5             # How much to boost low ball positions (0 = off)
```

### Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `enabled` | bool | false | Enable bounce weighting |
| `temporal_weight` | [0, 2+] | 1.0 | Strength of "late frames matter more" signal |
| `spatial_weight` | [0, 2+] | 0.5 | Strength of "low ball matters more" signal |

**Tuning Guide:**
- **temporal_weight=0**: Weighting disabled (equivalent to standard loss)
- **temporal_weight=1.0**: Moderate boost, frame N-1 worth ~1.5-2x frame 0
- **temporal_weight=2.0**: Strong boost, emphasizes bounce moment heavily
- **spatial_weight=0**: No spatial boost, use only temporal weighting
- **spatial_weight=0.5**: Moderate spatial boost, low ball 1.5-2x high ball
- **spatial_weight=1.0**: Strong spatial boost, coordinates matter as much as timing

## Implementation Details

### Data Flow

1. **Dataset (`ImageDataset`)**
   - Returns `(imgs, hms, xys, visis, num_frames)` during training
   - `num_frames`: number of frames in the clip [scalar per sample]
   - `xys`: ball positions [batch, num_frames, 2]

2. **Training Loop (`train_epoch`)**
   - Packs metadata into `frame_metadata` dict
   - Passes it to loss function

3. **Loss Function (`WeightedHeatmapLoss`)**
   - Checks dataset type (only applies to cricket)
   - Computes per-frame weights based on recency and position
   - Applies weights to targets, then computes base loss
   - Degrades gracefully if metadata unavailable

### Weight Computation

```python
# Frame indices [0, 1, 2, ..., N-1]
temporal_weight = (indices + 1) / N  # [1/N, 2/N, ..., 1]

# If ball positions available (normalized y ∈ [0,1])
spatial_bonus = 1 + spatial_weight * y_position

combined_weight = temporal_weight * spatial_bonus

# Normalize to mean=1 (modulate, don't scale)
normalized_weight = combined_weight / mean(combined_weight)
```

## Testing & Validation

### 1. Check It's Active
Run training with cricket config:
```bash
python src/main.py --config-name train_cricket
```

Monitor logs for:
```
INFO WeightedHeatmapLoss enabled: temporal_weight=1.0, spatial_weight=0.5
```

### 2. Verify Loss Is Applied
- Training loss should be **different** with/without weighting enabled
- Bounce frames (high y, late indices) should drive larger loss values
- Early frames should have diminished influence

### 3. Evaluate Quality
- Test F1/precision on **bounce frames specifically** (post-processing metric)
- Should improve relative to unweighted baseline
- Check that other frame types aren't degraded

### 4. Gradual Tuning
Start conservative, increase temporal_weight if convergence is slow:
```yaml
bounce_weight:
  enabled: true
  temporal_weight: 0.5    # Subtle (1.3x boost)
  spatial_weight: 0.25    # Minimal spatial
```

Then increase to focus harder on bounce:
```yaml
bounce_weight:
  enabled: true
  temporal_weight: 1.5    # Strong (2.5x boost)
  spatial_weight: 0.75    # Moderate spatial
```

## Impact on Other Sports

**Tennis, Volleyball, Badminton:** Unaffected. Config is cricket-only.

**If you want to enable for another sport:**
1. Add `bounce_weight` config section to that sport's YAML
2. Dataset names are in `configs/dataset/` (tennis.yaml, volleyball.yaml, etc.)
3. The loss function checks dataset type, so update accordingly

## Backwards Compatibility

### Old Code → New Code
- Old loss name: `heatmap` → New: `weighted_heatmap`
- If `bounce_weight` not in config, weighted loss behaves identically to base loss
- Existing trained models still work (loss is only during training)

### Disabling Weighting
Set `enabled: false` in config, or use standard `heatmap` loss name:
```yaml
loss:
  name: heatmap
  sub_name: mse
```

## Troubleshooting

### Loss goes to NaN
- Check that ground truth y-coordinates are valid (not all 0/invalid)
- Reduce `temporal_weight` (high weights can cause gradient issues early in training)
- Ensure heatmap targets are properly normalized [0, 1]

### No difference from unweighted
- Verify `enabled: true` in config and logs show activation
- Increase `temporal_weight` (current setting may be too subtle)
- Check that bounce frames actually have different y-positions in ground truth

### Loss diverges
- Reduce `temporal_weight` (e.g., 0.3 instead of 1.0)
- Disable spatial weighting temporarily (`spatial_weight: 0`)
- Verify learning rate is appropriate for weighted loss

## Advanced: Custom Weighting

To modify weighting behavior, edit `src/losses/weighted_heatmap.py`:

```python
# Change temporal weighting function (currently linear)
# Example: Exponential boost to final frames
temporal_weight = torch.exp(frame_indices / num_frames) / torch.exp(torch.tensor(1.0))

# Change spatial weighting (currently linear in y)
# Example: Quadratic boost to low ball
spatial_bonus = 1 + spatial_weight * (y_position ** 2)
```

Then rerun training—no config changes needed.
