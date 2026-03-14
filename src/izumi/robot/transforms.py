"""Action format conversions between model output and robot ServoCommand.

The model outputs 7D actions in the D405 camera optical frame:
    [dx, dy, dz, rx, ry, rz, gripper]

where:
  - (dx, dy, dz): translation delta in meters
  - (rx, ry, rz): rotation delta as axis-angle vector (rad)
  - gripper: absolute gripper opening in [0, 1]

ServoCommand expects a Pose3D delta in the EE (wrist_roll) frame with
quaternion orientation.  This module precomputes the constant rotation
matrix R_OPTICAL_IN_EE from the D405 URDF kinematic chain and applies it.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from stretch3_zmq.core.messages.pose_3d import Pose3D
from stretch3_zmq.core.messages.servo import ServoCommand
from stretch3_zmq.core.messages.vector_3d import Vector3D
from stretch3_zmq.core.messages.vector_4d import Vector4D


def _build_r_optical_in_ee() -> np.ndarray:
    """Compose the URDF chain to get R that maps vectors from optical to EE frame.

    Chain from d405_chain.urdf (link_wrist_roll → gripper_camera_color_optical_frame):
        T1: rpy=[0, 0, -π]
        T2: rpy=[0, -1.3963, -π/2]
        T3, T4: identity
        T5: rpy=[-π/2, 0, -π/2]

    Returns:
        (3, 3) rotation matrix R such that v_ee = R @ v_optical.
    """
    # Each URDF joint rotation: R = Rz(yaw)*Ry(pitch)*Rx(roll) = from_euler('XYZ', [r,p,y])
    r1 = Rotation.from_euler("XYZ", [0.0, 0.0, -np.pi])
    r2 = Rotation.from_euler("XYZ", [0.0, -1.3963, -np.pi / 2])
    r5 = Rotation.from_euler("XYZ", [-np.pi / 2, 0.0, -np.pi / 2])
    return (r1 * r2 * r5).as_matrix()


# Precomputed constant: transforms vectors from D405 optical frame → EE (wrist_roll) frame
R_OPTICAL_IN_EE: np.ndarray = _build_r_optical_in_ee()


def model_action_to_servo(action: np.ndarray) -> ServoCommand:
    """Convert a 7D model action to a ServoCommand.

    Args:
        action: ``(7,)`` float array ``[dx, dy, dz, rx, ry, rz, gripper]``
                expressed in the D405 camera optical frame.

    Returns:
        ``ServoCommand`` with ``ee_pose`` expressed in the EE (wrist_roll) frame.
    """
    t_optical = action[:3]
    aa_optical = action[3:6]
    gripper = float(np.clip(action[6], 0.0, 1.0))

    # Rotate translation and rotation axis from optical frame to EE frame
    t_ee = R_OPTICAL_IN_EE @ t_optical
    aa_ee = R_OPTICAL_IN_EE @ aa_optical

    # Axis-angle → quaternion (x, y, z, w)
    q = Rotation.from_rotvec(aa_ee).as_quat()

    return ServoCommand(
        ee_pose=Pose3D(
            position=Vector3D(x=float(t_ee[0]), y=float(t_ee[1]), z=float(t_ee[2])),
            orientation=Vector4D(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        ),
        gripper=gripper,
    )
