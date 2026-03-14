"""End-to-end inference pipeline connecting a policy model to a robot client.

The pipeline:
  1. Gets an RGB frame from the robot camera.
  2. Preprocesses it (HWC uint8 → CHW float in [0,1]).
  3. Pushes it into a rolling ``ObservationBuffer``.
  4. Runs the policy on the buffered image sequence.
  5. Converts the 7D action to a ``ServoCommand`` and sends it to the robot.
"""

import logging
import time

import numpy as np
import torch

from izumi.inference.buffer import ObservationBuffer
from izumi.inference.preprocessor import preprocess_frame
from izumi.robot.transforms import model_action_to_servo

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference loop: camera → model → robot.

    Both ``VQBeTPolicy`` and ``DiffusionPolicy`` satisfy the required interface
    (a ``step(images: Tensor) -> Tensor`` method).

    Usage::

        pipeline = InferencePipeline(policy, robot_client, buffer_size=3)
        pipeline.run(max_steps=200)
    """

    def __init__(
        self,
        policy,
        robot_client,
        buffer_size: int | None = None,
        control_hz: float = 5.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.robot_client = robot_client
        if buffer_size is None:
            buffer_size = getattr(policy, "obs_horizon", 3)
        self.buffer = ObservationBuffer(maxlen=buffer_size)
        self.control_hz = control_hz
        self.device = torch.device(device)

    def step(self) -> np.ndarray:
        """Run one inference step.

        Reads a camera frame, updates the observation buffer, runs the policy,
        converts the action to a ``ServoCommand``, and sends it to the robot.

        Returns:
            ``(7,)`` float32 numpy array ``[dx, dy, dz, rx, ry, rz, gripper]``
            in the camera optical frame (before frame transform).
        """
        frame = self.robot_client.get_camera_frame()  # (256, 256, 3) uint8
        tensor = preprocess_frame(frame)  # (3, 256, 256) float
        self.buffer.push(tensor)

        images = self.buffer.get_tensor().to(self.device)  # (T, 3, 256, 256)
        action_tensor = self.policy.step(images)  # (7,)
        action = action_tensor.cpu().numpy()  # (7,)

        servo = model_action_to_servo(action)
        self.robot_client.send_servo(servo)

        return action

    def run(self, max_steps: int | None = None) -> None:
        """Run the inference loop at the configured control frequency.

        Args:
            max_steps: Stop after this many steps.  ``None`` runs until
                       ``KeyboardInterrupt``.
        """
        period = 1.0 / self.control_hz
        step_count = 0

        logger.info("Starting inference loop at %.1f Hz", self.control_hz)
        try:
            while max_steps is None or step_count < max_steps:
                t_start = time.monotonic()

                self.step()
                step_count += 1

                elapsed = time.monotonic() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Inference loop stopped by user")

        logger.info("Inference loop finished after %d steps", step_count)
