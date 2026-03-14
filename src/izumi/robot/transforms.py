"""Action format conversions between model output and robot ServoCommand.

The model outputs 7D actions in the "personal camera" frame:
    [dx, dy, dz, rx, ry, rz, gripper]

where:
  - (dx, dy, dz): translation delta in meters
  - (rx, ry, rz): rotation delta as axis-angle vector (rad)
  - gripper: absolute gripper opening in [0, 1]

The "personal camera" frame is defined by the data-collection pipeline
(robot-utility-models/data-collection/process_from_r3ds.py): raw Record3D
poses are converted via ``apply_permutation_transform`` (P @ M @ P.T) before
being used as training labels.  At inference time the model therefore outputs
in the permuted frame:

    v_optical = P_R.T @ v_personal        (undo the permutation)
    v_ee      = R_OPTICAL_IN_EE @ v_optical

Combined: v_ee = R_OPTICAL_IN_EE @ P_R.T @ v_personal = R_MODEL_TO_EE @ v_model

where P_R is the 3x3 rotation block of the permutation matrix P from
``action_transforms.py``, and R_OPTICAL_IN_EE is derived from the D405
URDF kinematic chain (link_wrist_roll → gripper_camera_color_optical_frame).

ServoCommand expects a Pose3D delta in the EE (wrist_roll) frame with
quaternion orientation.
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
    r1 = Rotation.from_euler("XYZ", [0.0, 0.0, -np.pi])
    r2 = Rotation.from_euler("XYZ", [0.0, -1.3963, -np.pi / 2])
    r5 = Rotation.from_euler("XYZ", [-np.pi / 2, 0.0, -np.pi / 2])
    return (r1 * r2 * r5).as_matrix()


# Permutation matrix P_R from robot-utility-models/data-collection/utils/action_transforms.py.
# P_R maps vectors from D405 optical frame → "personal camera" (the model's training frame).
# P_R.T is the inverse: personal camera → D405 optical.
_P_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)

# R_OPTICAL_IN_EE: D405 optical frame → EE (wrist_roll) frame
R_OPTICAL_IN_EE: np.ndarray = _build_r_optical_in_ee()

# Full model-to-EE transform:
#   personal cam → D405 optical (P_R.T) → EE (R_OPTICAL_IN_EE)
R_MODEL_TO_EE: np.ndarray = R_OPTICAL_IN_EE @ _P_R.T


def model_action_to_servo(action: np.ndarray) -> ServoCommand:
    """Convert a 7D model action to a ServoCommand.

    Args:
        action: ``(7,)`` float array ``[dx, dy, dz, rx, ry, rz, gripper]``
                expressed in the model's "personal camera" frame.

    Returns:
        ``ServoCommand`` with ``ee_pose`` expressed in the EE (wrist_roll) frame.
    """
    t_model = action[:3]
    aa_model = action[3:6]
    gripper = float(np.clip(action[6], 0.0, 1.0))

    # Rotate translation and rotation axis from personal-camera frame to EE frame
    t_ee = R_MODEL_TO_EE @ t_model
    aa_ee = R_MODEL_TO_EE @ aa_model

    # Axis-angle → quaternion (x, y, z, w)
    q = Rotation.from_rotvec(aa_ee).as_quat()

    return ServoCommand(
        ee_pose=Pose3D(
            position=Vector3D(x=float(t_ee[0]), y=float(t_ee[1]), z=float(t_ee[2])),
            orientation=Vector4D(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        ),
        gripper=gripper,
    )
