"""Live robot integration tests.

These tests require a running stretch3-zmq driver and a reachable robot.

Running
-------
# All live tests (needs ROBOT_HOST set):
    ROBOT_HOST=192.168.1.100 uv run pytest tests/integration/test_robot_live.py -m robot -v

# Also run policy inference tests (needs checkpoint files):
    ROBOT_HOST=192.168.1.100 \\
    IZUMI_VQBET_CHECKPOINT=checkpoints/vqbet/door_opening/checkpoint.pt \\
    IZUMI_DIFFUSION_CHECKPOINT=checkpoints/diffusion/reorientation/checkpoint.pt \\
        uv run pytest tests/integration/test_robot_live.py -m robot -v

Safety
------
Steps 2-5 are safe with the runstop engaged.
Steps 6-7 do NOT send motion commands — they only verify inference runs without error.
"""

import os

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROBOT_HOST = os.environ.get("ROBOT_HOST", "")


def _require_robot():
    if not ROBOT_HOST:
        pytest.skip("ROBOT_HOST not set")


@pytest.fixture(scope="module")
def client():
    """Connected RobotClient for the whole module (one ZMQ connection)."""
    _require_robot()
    from izumi.robot.client import RobotClient

    with RobotClient(ROBOT_HOST) as c:
        yield c


# ---------------------------------------------------------------------------
# Step 2: ZMQ status stream
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_status_stream(client):
    """Smoke-test: receive one status message from the robot."""
    from stretch3_zmq.core.messages.status import Status

    status = client.get_status()
    assert isinstance(status, Status)
    assert status.joint_positions is not None


@pytest.mark.robot
def test_status_has_joint_positions(client):
    status = client.get_status()
    joints = status.joint_positions
    # joint_positions should be a non-empty mapping / object
    assert joints is not None


# ---------------------------------------------------------------------------
# Step 3: D405 camera
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_camera_frame_shape(client):
    """Camera frame must be (256, 256, 3) uint8."""
    frame = client.get_camera_frame()
    assert frame.shape == (256, 256, 3), f"unexpected shape: {frame.shape}"
    assert frame.dtype == np.uint8


@pytest.mark.robot
def test_camera_frame_not_black(client):
    """Frame should not be entirely zero (black)."""
    frame = client.get_camera_frame()
    assert frame.max() > 0, "camera frame is entirely black"


# ---------------------------------------------------------------------------
# Step 4: Action transform (no robot connection needed, but grouped here)
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_zero_action_servo_command(client):
    """Zero action → zero translation, identity quaternion."""
    from izumi.robot.transforms import model_action_to_servo

    cmd = model_action_to_servo(np.zeros(7))
    pos = cmd.ee_pose.position
    assert abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
    ori = cmd.ee_pose.orientation
    assert abs(ori.w - 1.0) < 1e-5, "expected identity quaternion"


@pytest.mark.robot
def test_nudge_action_servo_command(client):
    """Small Z nudge in camera frame should produce non-zero EE translation."""
    from izumi.robot.transforms import model_action_to_servo

    nudge = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5])
    cmd = model_action_to_servo(nudge)
    pos = cmd.ee_pose.position
    # At least one component must be non-zero
    assert abs(pos.x) + abs(pos.y) + abs(pos.z) > 1e-6


# ---------------------------------------------------------------------------
# Step 5: Send a zero servo command (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_send_zero_servo_command(client):
    """Send a zero servo command; robot should not move (zero delta)."""
    from izumi.robot.transforms import model_action_to_servo

    zero_cmd = model_action_to_servo(np.zeros(7))
    zero_cmd = zero_cmd.model_copy(update={"gripper": 0.5})
    client.send_servo(zero_cmd)  # raises on error


# ---------------------------------------------------------------------------
# Step 6: End-to-end VQ-BeT policy inference
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_vqbet_policy_inference(client):
    """Run one VQ-BeT policy step; verify output shape. Does NOT send commands."""
    checkpoint = os.environ.get("IZUMI_VQBET_CHECKPOINT", "")
    if not checkpoint:
        pytest.skip("IZUMI_VQBET_CHECKPOINT not set")

    from izumi.models.vqbet import VQBeTPolicy

    policy = VQBeTPolicy.from_checkpoint(checkpoint)

    frames = torch.stack(
        [torch.from_numpy(client.get_camera_frame()).float() / 255.0 for _ in range(3)]
    )  # (3, 256, 256, 3)
    frames = frames.permute(0, 3, 1, 2)  # (3, 3, 256, 256)

    action = policy.step(frames).numpy()
    assert action.shape == (7,), f"unexpected action shape: {action.shape}"
    assert np.isfinite(action).all(), "action contains non-finite values"


# ---------------------------------------------------------------------------
# Step 7: End-to-end Diffusion Policy inference
# ---------------------------------------------------------------------------


@pytest.mark.robot
def test_diffusion_policy_inference(client):
    """Run one Diffusion Policy step; verify output shape. Does NOT send commands."""
    checkpoint = os.environ.get("IZUMI_DIFFUSION_CHECKPOINT", "")
    if not checkpoint:
        pytest.skip("IZUMI_DIFFUSION_CHECKPOINT not set")

    from izumi.models.diffusion_policy import DiffusionPolicy

    policy = DiffusionPolicy.from_checkpoint(checkpoint)

    frames = torch.stack(
        [torch.from_numpy(client.get_camera_frame()).float() / 255.0 for _ in range(6)]
    )  # (6, 256, 256, 3)
    frames = frames.permute(0, 3, 1, 2)  # (6, 3, 256, 256)

    action = policy.step(frames).numpy()
    assert action.shape == (7,), f"unexpected action shape: {action.shape}"
    assert np.isfinite(action).all(), "action contains non-finite values"
