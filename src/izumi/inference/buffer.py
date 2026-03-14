"""Rolling observation buffer for the inference pipeline.

Maintains a fixed-length deque of preprocessed image tensors.  On the first
``push``, the buffer is padded with copies of that frame so subsequent
``get_tensor`` calls always return a full window.
"""

from collections import deque

import torch


class ObservationBuffer:
    """Fixed-length rolling buffer of image tensors.

    Usage::

        buf = ObservationBuffer(maxlen=3)
        buf.push(frame_tensor)          # (C, H, W) float
        images = buf.get_tensor()       # (T, C, H, W) float, T == maxlen
    """

    def __init__(self, maxlen: int = 3) -> None:
        self.maxlen = maxlen
        self._frames: deque[torch.Tensor] = deque(maxlen=maxlen)

    def push(self, frame: torch.Tensor) -> None:
        """Append a frame; if the buffer is empty, pad it to ``maxlen`` first.

        Args:
            frame: ``(C, H, W)`` float tensor.
        """
        if len(self._frames) == 0:
            for _ in range(self.maxlen):
                self._frames.append(frame)
        else:
            self._frames.append(frame)

    def get_tensor(self) -> torch.Tensor:
        """Return all buffered frames stacked along the time axis.

        Returns:
            ``(T, C, H, W)`` float tensor, where ``T == maxlen``.
        """
        return torch.stack(list(self._frames))

    def __len__(self) -> int:
        return len(self._frames)
