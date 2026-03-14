"""Unit tests for robot action format conversions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stretch3_zmq.core.messages.servo import ServoCommand

from izumi.robot.transforms import R_MODEL_TO_EE, R_OPTICAL_IN_EE, model_action_to_servo


def test_r_optical_in_ee_is_rotation_matrix():
    """R must be a proper rotation matrix (det=1, R^T R = I)."""
    R = R_OPTICAL_IN_EE
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_zero_action_gives_zero_translation():
    action = np.zeros(7)
    cmd = model_action_to_servo(action)
    assert cmd.ee_pose.position.x == pytest.approx(0.0)
    assert cmd.ee_pose.position.y == pytest.approx(0.0)
    assert cmd.ee_pose.position.z == pytest.approx(0.0)


def test_zero_action_gives_identity_quaternion():
    action = np.zeros(7)
    cmd = model_action_to_servo(action)
    # Identity rotation: (x,y,z,w) = (0,0,0,1)
    assert cmd.ee_pose.orientation.x == pytest.approx(0.0, abs=1e-7)
    assert cmd.ee_pose.orientation.y == pytest.approx(0.0, abs=1e-7)
    assert cmd.ee_pose.orientation.z == pytest.approx(0.0, abs=1e-7)
    assert cmd.ee_pose.orientation.w == pytest.approx(1.0, abs=1e-7)


def test_translation_rotated_by_r():
    """A pure translation action should be rotated by R_MODEL_TO_EE."""
    t_model = np.array([1.0, 0.0, 0.0])
    action = np.array([*t_model, 0.0, 0.0, 0.0, 0.5])
    cmd = model_action_to_servo(action)
    expected = R_MODEL_TO_EE @ t_model
    assert cmd.ee_pose.position.x == pytest.approx(float(expected[0]))
    assert cmd.ee_pose.position.y == pytest.approx(float(expected[1]))
    assert cmd.ee_pose.position.z == pytest.approx(float(expected[2]))


def test_known_rotation_produces_unit_quaternion():
    """Quaternion from any rotation should be a unit quaternion."""
    action = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5])
    cmd = model_action_to_servo(action)
    o = cmd.ee_pose.orientation
    norm = (o.x**2 + o.y**2 + o.z**2 + o.w**2) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_gripper_passthrough():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    cmd = model_action_to_servo(action)
    assert cmd.gripper == pytest.approx(0.7)


def test_gripper_clamp_above_one():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])
    cmd = model_action_to_servo(action)
    assert cmd.gripper == pytest.approx(1.0)


def test_gripper_clamp_below_zero():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3])
    cmd = model_action_to_servo(action)
    assert cmd.gripper == pytest.approx(0.0)


def test_output_type_is_servo_command():
    action = np.zeros(7)
    cmd = model_action_to_servo(action)
    assert isinstance(cmd, ServoCommand)


def test_rotation_roundtrip():
    """Axis-angle → quaternion → back to axis-angle should match R_MODEL_TO_EE applied to input."""
    aa_model = np.array([0.05, 0.03, -0.02])
    action = np.array([0.0, 0.0, 0.0, *aa_model, 0.5])
    cmd = model_action_to_servo(action)
    o = cmd.ee_pose.orientation
    q = np.array([o.x, o.y, o.z, o.w])
    aa_recovered_ee = Rotation.from_quat(q).as_rotvec()
    aa_expected_ee = R_MODEL_TO_EE @ aa_model
    np.testing.assert_allclose(aa_recovered_ee, aa_expected_ee, atol=1e-6)
