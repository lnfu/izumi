"""Smoke tests to verify project scaffold is working."""


def test_izumi_importable():
    import izumi

    assert izumi is not None


def test_stretch3_zmq_core_importable():
    from stretch3_zmq.core.messages.pose_3d import Pose3D
    from stretch3_zmq.core.messages.servo import ServoCommand

    cmd = ServoCommand(ee_pose=Pose3D(), gripper=0.5)
    assert cmd.gripper == 0.5


def test_torch_importable():
    import torch

    assert torch.cuda.is_available() or True  # passes even without GPU
