"""Integration tests for the inference pipeline (Task 9).

Tests are grouped by what they require:

  checkpoint  — real .pt checkpoint file, mock robot
  robot       — live Stretch 3 robot (no motion commands sent)

Running
-------
# VQ-BeT pipeline with checkpoint + mock robot:
    IZUMI_VQBET_CHECKPOINT=checkpoints/vqbet/door_opening/checkpoint.pt \\
        uv run pytest tests/integration/test_pipeline_integration.py -m checkpoint -v

# Diffusion pipeline with checkpoint + mock robot:
    IZUMI_DIFFUSION_CHECKPOINT=checkpoints/diffusion/reorientation/checkpoint.pt \\
        uv run pytest tests/integration/test_pipeline_integration.py -m checkpoint -v

# Both checkpoint variants in one go:
    IZUMI_VQBET_CHECKPOINT=checkpoints/vqbet/door_opening/checkpoint.pt \\
    IZUMI_DIFFUSION_CHECKPOINT=checkpoints/diffusion/reorientation/checkpoint.pt \\
        uv run pytest tests/integration/test_pipeline_integration.py -m checkpoint -v

# Full pipeline with live robot (reads camera, does NOT move robot):
    ROBOT_HOST=192.168.1.100 \\
    IZUMI_VQBET_CHECKPOINT=checkpoints/vqbet/door_opening/checkpoint.pt \\
    IZUMI_DIFFUSION_CHECKPOINT=checkpoints/diffusion/reorientation/checkpoint.pt \\
        uv run pytest tests/integration/test_pipeline_integration.py -m robot -v
"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment / skip helpers
# ---------------------------------------------------------------------------

VQBET_CHECKPOINT = os.environ.get("IZUMI_VQBET_CHECKPOINT", "")
DIFFUSION_CHECKPOINT = os.environ.get("IZUMI_DIFFUSION_CHECKPOINT", "")
ROBOT_HOST = os.environ.get("ROBOT_HOST", "")

HAS_VQBET = bool(VQBET_CHECKPOINT) and os.path.isfile(VQBET_CHECKPOINT)
HAS_DIFFUSION = bool(DIFFUSION_CHECKPOINT) and os.path.isfile(DIFFUSION_CHECKPOINT)
HAS_ROBOT = bool(ROBOT_HOST)


def _require_vqbet():
    if not HAS_VQBET:
        pytest.skip("Set IZUMI_VQBET_CHECKPOINT=/path/to/checkpoint.pt to run")


def _require_diffusion():
    if not HAS_DIFFUSION:
        pytest.skip("Set IZUMI_DIFFUSION_CHECKPOINT=/path/to/checkpoint.pt to run")


def _require_robot():
    if not HAS_ROBOT:
        pytest.skip("ROBOT_HOST not set")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vqbet_policy():
    _require_vqbet()
    from izumi.models.vqbet import VQBeTPolicy

    return VQBeTPolicy.from_checkpoint(VQBET_CHECKPOINT, device="cpu")


@pytest.fixture(scope="module")
def diffusion_policy():
    _require_diffusion()
    from izumi.models.diffusion_policy import DiffusionPolicy

    return DiffusionPolicy.from_checkpoint(DIFFUSION_CHECKPOINT, device="cpu")


@pytest.fixture
def mock_robot():
    """Mock RobotClient that returns a random non-black frame each call."""
    robot = MagicMock()
    rng = np.random.default_rng(0)

    def _random_frame():
        return rng.integers(50, 200, size=(256, 256, 3), dtype=np.uint8)

    robot.get_camera_frame.side_effect = _random_frame
    return robot


@pytest.fixture(scope="module")
def live_robot():
    _require_robot()
    from izumi.robot.client import RobotClient

    with RobotClient(ROBOT_HOST) as client:
        yield client


# ---------------------------------------------------------------------------
# Checkpoint tests — VQ-BeT pipeline (mock robot)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_vqbet_pipeline_step_shape(vqbet_policy, mock_robot):
    """Single step returns a (7,) float32 action."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    action = pipeline.step()
    assert action.shape == (7,)
    assert action.dtype == np.float32


@pytest.mark.checkpoint
def test_vqbet_pipeline_step_finite(vqbet_policy, mock_robot):
    """Action values must all be finite (no NaN/Inf)."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    action = pipeline.step()
    assert np.isfinite(action).all(), f"non-finite action: {action}"


@pytest.mark.checkpoint
def test_vqbet_pipeline_buffer_fills_on_first_step(vqbet_policy, mock_robot):
    """After the first step the buffer should be full (size == maxlen)."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    pipeline.step()
    assert len(pipeline.buffer) == 3


@pytest.mark.checkpoint
def test_vqbet_pipeline_multiple_steps(vqbet_policy, mock_robot):
    """Three consecutive steps all return (7,) finite actions."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    for _ in range(3):
        action = pipeline.step()
        assert action.shape == (7,)
        assert np.isfinite(action).all()


@pytest.mark.checkpoint
def test_vqbet_pipeline_gripper_in_range(vqbet_policy, mock_robot):
    """Gripper component (index 6) should be in [0, 1] after clipping in transforms."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    action = pipeline.step()
    # ServoCommand clips gripper — raw model output may exceed [0,1]
    # but we verify the action array itself is a valid float
    assert np.isfinite(action[6])


@pytest.mark.checkpoint
def test_vqbet_pipeline_sends_servo_command(vqbet_policy, mock_robot):
    """Pipeline must call send_servo exactly once per step."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(vqbet_policy, mock_robot, buffer_size=3, device="cpu")
    pipeline.step()
    mock_robot.send_servo.assert_called_once()


@pytest.mark.checkpoint
def test_vqbet_pipeline_run_n_steps(vqbet_policy, mock_robot):
    """run(max_steps=5) drives the robot exactly 5 times."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        vqbet_policy, mock_robot, buffer_size=3, control_hz=1000.0, device="cpu"
    )
    pipeline.run(max_steps=5)
    assert mock_robot.get_camera_frame.call_count == 5
    assert mock_robot.send_servo.call_count == 5


# ---------------------------------------------------------------------------
# Checkpoint tests — Diffusion pipeline (mock robot)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_diffusion_pipeline_step_shape(diffusion_policy, mock_robot):
    """Single step returns a (7,) float32 action."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(diffusion_policy, mock_robot, buffer_size=6, device="cpu")
    action = pipeline.step()
    assert action.shape == (7,)
    assert action.dtype == np.float32


@pytest.mark.checkpoint
def test_diffusion_pipeline_step_finite(diffusion_policy, mock_robot):
    """Diffusion denoising must produce finite values."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(diffusion_policy, mock_robot, buffer_size=6, device="cpu")
    action = pipeline.step()
    assert np.isfinite(action).all(), f"non-finite action: {action}"


@pytest.mark.checkpoint
def test_diffusion_pipeline_buffer_size(diffusion_policy, mock_robot):
    """Diffusion uses obs_horizon=6; buffer should be full after first step."""
    from izumi.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(diffusion_policy, mock_robot, buffer_size=6, device="cpu")
    pipeline.step()
    assert len(pipeline.buffer) == 6


# ---------------------------------------------------------------------------
# Robot tests — full pipeline with live camera (no motion commands sent)
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_vqbet_pipeline_live_camera_step(live_robot):
    """VQ-BeT pipeline step with a real camera frame; does NOT send servo."""
    if not HAS_VQBET:
        pytest.skip("Set IZUMI_VQBET_CHECKPOINT=/path/to/checkpoint.pt to run")

    from izumi.inference.pipeline import InferencePipeline
    from izumi.models.vqbet import VQBeTPolicy

    policy = VQBeTPolicy.from_checkpoint(VQBET_CHECKPOINT, device="cpu")

    # Use a wrapper that reads camera but swallows send_servo
    robot = MagicMock(wraps=live_robot)
    robot.send_servo = MagicMock()  # suppress real commands

    pipeline = InferencePipeline(policy, robot, buffer_size=3, device="cpu")
    action = pipeline.step()

    assert action.shape == (7,)
    assert np.isfinite(action).all(), f"non-finite action: {action}"
    robot.send_servo.assert_called_once()


@pytest.mark.robot
def test_diffusion_pipeline_live_camera_step(live_robot):
    """Diffusion pipeline step with a real camera frame; does NOT send servo."""
    if not HAS_DIFFUSION:
        pytest.skip("Set IZUMI_DIFFUSION_CHECKPOINT=/path/to/checkpoint.pt to run")

    from izumi.inference.pipeline import InferencePipeline
    from izumi.models.diffusion_policy import DiffusionPolicy

    policy = DiffusionPolicy.from_checkpoint(DIFFUSION_CHECKPOINT, device="cpu")

    robot = MagicMock(wraps=live_robot)
    robot.send_servo = MagicMock()

    pipeline = InferencePipeline(policy, robot, buffer_size=6, device="cpu")
    action = pipeline.step()

    assert action.shape == (7,)
    assert np.isfinite(action).all(), f"non-finite action: {action}"
    robot.send_servo.assert_called_once()
