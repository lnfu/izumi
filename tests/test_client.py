"""Unit tests for RobotClient using mocked ZMQ sockets."""

import struct
import time
from unittest.mock import MagicMock, patch

import blosc2
import numpy as np
import pytest
from stretch3_zmq.core.messages.command import ManipulatorCommand
from stretch3_zmq.core.messages.orientation import Orientation
from stretch3_zmq.core.messages.pose_2d import Pose2D
from stretch3_zmq.core.messages.pose_3d import Pose3D
from stretch3_zmq.core.messages.servo import ServoCommand
from stretch3_zmq.core.messages.status import IMU, Odometry, Status
from stretch3_zmq.core.messages.twist_2d import Twist2D
from stretch3_zmq.core.messages.vector_3d import Vector3D
from stretch3_zmq.core.messages.vector_4d import Vector4D

from izumi.robot.client import _D405_HEIGHT, _D405_WIDTH, TARGET_SIZE, RobotClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timestamp_bytes() -> bytes:
    return struct.pack("!Q", time.time_ns())


def _make_dummy_status() -> Status:
    return Status(
        is_charging=False,
        is_low_voltage=False,
        runstop=False,
        odometry=Odometry(pose=Pose2D(x=0, y=0, theta=0), twist=Twist2D(vx=0, wz=0)),
        imu=IMU(
            orientation=Orientation(roll=0, pitch=0, yaw=0),
            acceleration=Vector3D(),
            gyro=Vector3D(),
        ),
        joint_positions=tuple([0.0] * 10),
        joint_velocities=tuple([0.0] * 10),
        joint_efforts=tuple([0.0] * 10),
    )


def _make_dummy_servo() -> ServoCommand:
    return ServoCommand(
        ee_pose=Pose3D(position=Vector3D(), orientation=Vector4D()),
        gripper=0.5,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return RobotClient("test-robot", camera_timeout_ms=100, status_timeout_ms=100)


@pytest.fixture
def connected_client():
    """RobotClient with mocked ZMQ sockets."""
    c = RobotClient("test-robot", camera_timeout_ms=100, status_timeout_ms=100)
    mock_ctx = MagicMock()

    status_sock = MagicMock()
    d405_sock = MagicMock()
    servo_sock = MagicMock()
    command_sock = MagicMock()

    def socket_factory(sock_type):
        import zmq

        if sock_type == zmq.SUB:
            return [status_sock, d405_sock].pop(0)
        return [servo_sock, command_sock].pop(0)

    mock_ctx.socket.side_effect = socket_factory

    with patch("izumi.robot.client.zmq.Context", return_value=mock_ctx):
        c.connect()

    c._status_sub = status_sock
    c._d405_sub = d405_sock
    c._servo_pub = servo_sock
    c._command_pub = command_sock
    return c


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_default_ports(client):
    assert client.ports["status"] == 5555
    assert client.ports["servo"] == 5558
    assert client.ports["d405"] == 6002
    assert client.ports["command"] == 5556


def test_custom_port_override():
    c = RobotClient("robot", ports={"servo": 9999})
    assert c.ports["servo"] == 9999
    assert c.ports["status"] == 5555  # others stay default


def test_not_connected_raises_on_camera(client):
    with pytest.raises(RuntimeError, match="connect"):
        client.get_camera_frame()


def test_not_connected_raises_on_status(client):
    with pytest.raises(RuntimeError, match="connect"):
        client.get_status()


def test_not_connected_raises_on_servo(client):
    with pytest.raises(RuntimeError, match="connect"):
        client.send_servo(_make_dummy_servo())


def test_camera_timeout(connected_client):
    connected_client._d405_sub.poll.return_value = 0  # no message
    with pytest.raises(TimeoutError):
        connected_client.get_camera_frame()


def test_status_timeout(connected_client):
    connected_client._status_sub.poll.return_value = 0
    with pytest.raises(TimeoutError):
        connected_client.get_status()


def test_get_camera_frame_shape(connected_client):
    # Build a fake blosc2-compressed RGB frame
    frame = np.zeros((_D405_HEIGHT, _D405_WIDTH, 3), dtype=np.uint8)
    payload = blosc2.compress(frame.tobytes(), typesize=1)
    ts = _make_timestamp_bytes()

    connected_client._d405_sub.poll.return_value = 1
    connected_client._d405_sub.recv_multipart.return_value = [b"rgb", ts, payload]

    result = connected_client.get_camera_frame()
    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)
    assert result.dtype == np.uint8


def test_get_camera_frame_raw_fallback(connected_client):
    """Client falls back to raw bytes if blosc2 decompression fails."""
    frame = np.zeros((_D405_HEIGHT, _D405_WIDTH, 3), dtype=np.uint8)
    ts = _make_timestamp_bytes()

    connected_client._d405_sub.poll.return_value = 1
    connected_client._d405_sub.recv_multipart.return_value = [b"rgb", ts, frame.tobytes()]

    result = connected_client.get_camera_frame()
    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)


def test_get_status_returns_status_object(connected_client):
    status = _make_dummy_status()
    payload = status.to_bytes()
    ts = _make_timestamp_bytes()

    connected_client._status_sub.poll.return_value = 1
    connected_client._status_sub.recv_multipart.return_value = [ts, payload]

    result = connected_client.get_status()
    assert isinstance(result, Status)
    assert result.is_charging == status.is_charging


def test_send_servo_sends_three_frames(connected_client):
    cmd = _make_dummy_servo()
    connected_client.send_servo(cmd)

    assert connected_client._servo_pub.send_multipart.called
    sent = connected_client._servo_pub.send_multipart.call_args[0][0]
    assert sent[0] == b"servo"
    assert len(sent) == 3  # [topic, timestamp, payload]


def test_send_manipulator_sends_three_frames(connected_client):
    cmd = ManipulatorCommand(joint_positions=tuple([0.0] * 10))
    connected_client.send_manipulator(cmd)

    assert connected_client._command_pub.send_multipart.called
    sent = connected_client._command_pub.send_multipart.call_args[0][0]
    assert sent[0] == b"manipulator"
    assert len(sent) == 3


def test_disconnect_closes_sockets(connected_client):
    connected_client.disconnect()
    assert connected_client._status_sub is None
    assert connected_client._d405_sub is None
    assert connected_client._servo_pub is None
    assert connected_client._command_pub is None
    assert connected_client._ctx is None


def test_context_manager_connects_and_disconnects():
    with patch("izumi.robot.client.zmq.Context") as mock_ctx_cls:
        mock_ctx = MagicMock()
        mock_ctx_cls.return_value = mock_ctx
        mock_ctx.socket.return_value = MagicMock()

        with RobotClient("robot") as c:
            assert c._ctx is not None

        assert c._ctx is None
