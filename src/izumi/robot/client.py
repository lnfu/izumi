"""ZMQ client for stretch3-zmq robot driver.

Connects to the driver running on the robot and provides:
  - Camera frames (D405 RGB, port 6002)
  - Robot status (port 5555)
  - Servo commands (EE-space delta, port 5558)
  - Manipulator commands (joint positions, port 5556)

Protocol (driver always binds, client always connects):
  - Status:     driver PUB binds  → client SUB connects
  - D405 camera: driver PUB binds  → client SUB connects (topic b"rgb")
  - Servo:      driver SUB binds  → client PUB connects (topic "servo")
  - Command:    driver SUB binds  → client PUB connects (topics "manipulator", "base")
"""

import logging
import os

import blosc2
import cv2
import numpy as np
import zmq
from stretch3_zmq.core.messages.command import ManipulatorCommand
from stretch3_zmq.core.messages.protocol import decode_with_timestamp, encode_with_timestamp
from stretch3_zmq.core.messages.servo import ServoCommand
from stretch3_zmq.core.messages.status import Status

logger = logging.getLogger(__name__)

# Default ports matching DriverConfig
_DEFAULT_PORTS: dict[str, int] = {
    "status": 5555,
    "command": 5556,
    "servo": 5558,
    "d405": 6002,
}

# D405 native resolution
_D405_HEIGHT = 480
_D405_WIDTH = 640
_D405_CHANNELS = 3

# Target resolution for the model
TARGET_SIZE = 256


class RobotClient:
    """ZMQ client for the stretch3-zmq robot driver.

    Usage::

        client = RobotClient("192.168.1.100")
        client.connect()
        frame = client.get_camera_frame()      # (256, 256, 3) uint8 RGB
        status = client.get_status()           # Status
        client.send_servo(servo_command)
        client.disconnect()

    Or as a context manager::

        with RobotClient("192.168.1.100") as client:
            frame = client.get_camera_frame()
    """

    def __init__(
        self,
        host: str,
        ports: dict[str, int] | None = None,
        camera_timeout_ms: int = 2000,
        status_timeout_ms: int = 2000,
    ) -> None:
        self.host = os.path.expandvars(host)
        self.ports = {**_DEFAULT_PORTS, **(ports or {})}
        self.camera_timeout_ms = camera_timeout_ms
        self.status_timeout_ms = status_timeout_ms

        self._ctx: zmq.Context | None = None
        self._status_sub: zmq.Socket | None = None
        self._d405_sub: zmq.Socket | None = None
        self._servo_pub: zmq.Socket | None = None
        self._command_pub: zmq.Socket | None = None

    def connect(self) -> None:
        """Establish all ZMQ connections to the robot driver."""
        self._ctx = zmq.Context()

        # Status subscriber (no topic filter — 2-frame multipart)
        self._status_sub = self._ctx.socket(zmq.SUB)
        self._status_sub.setsockopt(zmq.RCVHWM, 8)
        self._status_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._status_sub.connect(f"tcp://{self.host}:{self.ports['status']}")

        # D405 camera subscriber (topic "rgb" — 3-frame multipart)
        self._d405_sub = self._ctx.socket(zmq.SUB)
        self._d405_sub.setsockopt(zmq.RCVHWM, 8)
        self._d405_sub.setsockopt(zmq.SUBSCRIBE, b"rgb")
        self._d405_sub.connect(f"tcp://{self.host}:{self.ports['d405']}")

        # Servo command publisher (topic "servo" — 3-frame multipart)
        self._servo_pub = self._ctx.socket(zmq.PUB)
        self._servo_pub.setsockopt(zmq.SNDHWM, 8)
        self._servo_pub.connect(f"tcp://{self.host}:{self.ports['servo']}")

        # Manipulator command publisher (topic "manipulator")
        self._command_pub = self._ctx.socket(zmq.PUB)
        self._command_pub.setsockopt(zmq.SNDHWM, 8)
        self._command_pub.connect(f"tcp://{self.host}:{self.ports['command']}")

        logger.info(f"RobotClient connected to {self.host}")

    def disconnect(self) -> None:
        """Close all sockets and terminate the ZMQ context."""
        for sock in (self._status_sub, self._d405_sub, self._servo_pub, self._command_pub):
            if sock is not None:
                sock.close()
        self._status_sub = None
        self._d405_sub = None
        self._servo_pub = None
        self._command_pub = None

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        logger.info("RobotClient disconnected")

    def get_camera_frame(self) -> np.ndarray:
        """Receive one RGB frame from the D405 camera.

        Blocks until a frame arrives (up to ``camera_timeout_ms``).

        Returns:
            ``(TARGET_SIZE, TARGET_SIZE, 3)`` uint8 RGB array.

        Raises:
            TimeoutError: If no frame arrives within the timeout.
            RuntimeError: If not connected.
        """
        if self._d405_sub is None:
            raise RuntimeError("Not connected — call connect() first")

        if not self._d405_sub.poll(self.camera_timeout_ms):
            raise TimeoutError(f"No camera frame within {self.camera_timeout_ms} ms")

        parts = self._d405_sub.recv_multipart()
        # parts: [b"rgb", timestamp, payload]
        payload = parts[2]

        # Try blosc2 decompression; fall back to raw bytes
        try:
            raw = blosc2.decompress(payload)
        except Exception:
            raw = payload

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            _D405_HEIGHT, _D405_WIDTH, _D405_CHANNELS
        )
        return cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    def get_status(self) -> Status:
        """Receive the latest robot status.

        Blocks until a status message arrives (up to ``status_timeout_ms``).

        Returns:
            ``Status`` with joint positions, IMU, odometry, etc.

        Raises:
            TimeoutError: If no status arrives within the timeout.
            RuntimeError: If not connected.
        """
        if self._status_sub is None:
            raise RuntimeError("Not connected — call connect() first")

        if not self._status_sub.poll(self.status_timeout_ms):
            raise TimeoutError(f"No status within {self.status_timeout_ms} ms")

        parts = self._status_sub.recv_multipart()
        _timestamp_ns, payload = decode_with_timestamp(parts)
        return Status.from_bytes(payload)

    def send_servo(self, command: ServoCommand) -> None:
        """Send an EE-space servo command to the robot.

        Args:
            command: ``ServoCommand`` with delta pose and absolute gripper.

        Raises:
            RuntimeError: If not connected.
        """
        if self._servo_pub is None:
            raise RuntimeError("Not connected — call connect() first")

        parts = [b"servo", *encode_with_timestamp(command.to_bytes())]
        self._servo_pub.send_multipart(parts)

    def send_manipulator(self, command: ManipulatorCommand) -> None:
        """Send a joint-space manipulator command to the robot.

        Args:
            command: ``ManipulatorCommand`` with 10 joint positions.

        Raises:
            RuntimeError: If not connected.
        """
        if self._command_pub is None:
            raise RuntimeError("Not connected — call connect() first")

        parts = [b"manipulator", *encode_with_timestamp(command.to_bytes())]
        self._command_pub.send_multipart(parts)

    def __enter__(self) -> "RobotClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
