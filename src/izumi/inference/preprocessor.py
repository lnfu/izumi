"""Image preprocessing for the inference pipeline.

Converts raw camera frames (HWC uint8 RGB, already at target resolution)
into model-ready CHW float tensors in [0, 1].
"""

import numpy as np
import torch


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8 RGB frame to a CHW float32 tensor in [0, 1].

    The input is expected to already be at the target resolution (256x256)
    as returned by ``RobotClient.get_camera_frame()``.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        ``(3, H, W)`` float32 tensor with values in ``[0, 1]``.
    """
    return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
