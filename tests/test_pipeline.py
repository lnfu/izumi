"""Unit tests for the inference pipeline (buffer, preprocessor, pipeline)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from izumi.inference.buffer import ObservationBuffer
from izumi.inference.pipeline import InferencePipeline
from izumi.inference.preprocessor import preprocess_frame

# ---------------------------------------------------------------------------
# preprocess_frame
# ---------------------------------------------------------------------------


class TestPreprocessFrame:
    def test_output_shape(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        t = preprocess_frame(frame)
        assert t.shape == (3, 256, 256)

    def test_dtype_float(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        t = preprocess_frame(frame)
        assert t.dtype == torch.float32

    def test_zero_maps_to_zero(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        t = preprocess_frame(frame)
        assert t.sum().item() == pytest.approx(0.0)

    def test_255_maps_to_one(self):
        frame = np.full((256, 256, 3), 255, dtype=np.uint8)
        t = preprocess_frame(frame)
        assert t.max().item() == pytest.approx(1.0)

    def test_channel_order(self):
        # R=100, G=0, B=0 → channel 0 should be nonzero
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 100
        t = preprocess_frame(frame)
        assert t[0].sum().item() > 0.0
        assert t[1].sum().item() == pytest.approx(0.0)
        assert t[2].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ObservationBuffer
# ---------------------------------------------------------------------------


class TestObservationBuffer:
    def test_pads_on_first_push(self):
        buf = ObservationBuffer(maxlen=3)
        frame = torch.zeros(3, 4, 4)
        buf.push(frame)
        assert len(buf) == 3

    def test_pad_fills_with_same_frame(self):
        buf = ObservationBuffer(maxlen=3)
        frame = torch.ones(3, 4, 4) * 0.5
        buf.push(frame)
        tensor = buf.get_tensor()
        assert tensor.shape == (3, 3, 4, 4)
        assert (tensor == 0.5).all()

    def test_rolling_discards_oldest(self):
        buf = ObservationBuffer(maxlen=3)
        for i in range(5):
            buf.push(torch.full((3, 4, 4), float(i)))
        tensor = buf.get_tensor()
        assert tensor.shape == (3, 3, 4, 4)
        # frames 2, 3, 4 remain (0-indexed)
        assert tensor[0, 0, 0, 0].item() == pytest.approx(2.0)
        assert tensor[1, 0, 0, 0].item() == pytest.approx(3.0)
        assert tensor[2, 0, 0, 0].item() == pytest.approx(4.0)

    def test_get_tensor_shape(self):
        buf = ObservationBuffer(maxlen=5)
        buf.push(torch.zeros(3, 8, 8))
        assert buf.get_tensor().shape == (5, 3, 8, 8)

    def test_second_push_shifts_window(self):
        buf = ObservationBuffer(maxlen=3)
        buf.push(torch.zeros(3, 4, 4))  # fills [0, 0, 0]
        buf.push(torch.ones(3, 4, 4))  # becomes [0, 0, 1]
        tensor = buf.get_tensor()
        assert tensor[0, 0, 0, 0].item() == pytest.approx(0.0)
        assert tensor[2, 0, 0, 0].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


def _make_pipeline(buffer_size: int = 3, action_dim: int = 7):
    """Return a pipeline with mocked policy and robot client."""
    policy = MagicMock()
    policy.step.return_value = torch.zeros(action_dim)

    robot = MagicMock()
    robot.get_camera_frame.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

    pipeline = InferencePipeline(
        policy=policy,
        robot_client=robot,
        buffer_size=buffer_size,
        control_hz=100.0,  # fast for tests
        device="cpu",
    )
    return pipeline, policy, robot


class TestInferencePipeline:
    def test_step_returns_7d_action(self):
        pipeline, _, _ = _make_pipeline()
        action = pipeline.step()
        assert action.shape == (7,)

    def test_step_calls_camera_policy_servo(self):
        pipeline, policy, robot = _make_pipeline()
        pipeline.step()
        robot.get_camera_frame.assert_called_once()
        policy.step.assert_called_once()
        robot.send_servo.assert_called_once()

    def test_step_buffer_filled_on_first_call(self):
        pipeline, _, _ = _make_pipeline(buffer_size=3)
        pipeline.step()
        assert len(pipeline.buffer) == 3

    def test_step_policy_receives_correct_tensor_shape(self):
        pipeline, policy, _ = _make_pipeline(buffer_size=3)
        pipeline.step()
        images_arg = policy.step.call_args[0][0]
        assert images_arg.shape == (3, 3, 256, 256)

    def test_step_action_is_numpy_float32(self):
        pipeline, _, _ = _make_pipeline()
        action = pipeline.step()
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32

    def test_run_max_steps(self):
        pipeline, policy, robot = _make_pipeline()
        pipeline.run(max_steps=4)
        assert robot.get_camera_frame.call_count == 4
        assert policy.step.call_count == 4
        assert robot.send_servo.call_count == 4

    def test_run_zero_steps(self):
        pipeline, policy, _ = _make_pipeline()
        pipeline.run(max_steps=0)
        policy.step.assert_not_called()

    def test_different_buffer_sizes(self):
        for buf_size in (1, 3, 6):
            pipeline, policy, _ = _make_pipeline(buffer_size=buf_size)
            pipeline.step()
            images_arg = policy.step.call_args[0][0]
            assert images_arg.shape[0] == buf_size
