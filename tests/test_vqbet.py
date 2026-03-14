"""Tests for VQ-BeT policy.

Unit tests use random weights (no network, no checkpoint).
Checkpoint tests require IZUMI_CHECKPOINT pointing to a VQ-BeT checkpoint
AND network access (for dobb-e architecture from HuggingFace Hub).
"""

import os

import pytest
import torch

from izumi.models.gpt import GPTConfig
from izumi.models.vqbet import VQBehaviorTransformer, VQBeTPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("IZUMI_CHECKPOINT")
HAS_CHECKPOINT = CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def vqbet():
    m = VQBehaviorTransformer()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Unit tests — VQBehaviorTransformer (no network, no checkpoint)
# ---------------------------------------------------------------------------


def test_vqbet_step_output_shape(vqbet):
    cfg = GPTConfig()
    obs = torch.randn(3, cfg.output_dim)  # T=3, obs_dim=256
    with torch.no_grad():
        out = vqbet.step(obs)
    assert out.shape == (3, 1, 7)  # (T, input_dim_h, n_action)


def test_vqbet_step_output_finite(vqbet):
    cfg = GPTConfig()
    obs = torch.randn(3, cfg.output_dim)
    with torch.no_grad():
        out = vqbet.step(obs)
    assert torch.isfinite(out).all()


def test_vqbet_step_single_step(vqbet):
    cfg = GPTConfig()
    obs = torch.randn(1, cfg.output_dim)
    with torch.no_grad():
        out = vqbet.step(obs)
    assert out.shape == (1, 1, 7)


def test_vqbet_state_dict_roots(vqbet):
    roots = {k.split(".")[0] for k in vqbet.state_dict()}
    assert roots == {
        "_gpt_model",
        "_vqvae_model",
        "_map_to_cbet_preds_bin1",
        "_map_to_cbet_preds_bin2",
        "_map_to_cbet_preds_offset",
    }


# ---------------------------------------------------------------------------
# Checkpoint + hub tests (needs IZUMI_CHECKPOINT and HuggingFace network)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_loads_without_error():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    policy = VQBeTPolicy.from_checkpoint(CHECKPOINT_PATH)
    assert policy is not None


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_step_output_shape():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    policy = VQBeTPolicy.from_checkpoint(CHECKPOINT_PATH)
    images = torch.zeros(3, 3, 256, 256)  # (T=3, C, H, W)
    action = policy.step(images)
    assert action.shape == (7,)


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_step_output_finite():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    policy = VQBeTPolicy.from_checkpoint(CHECKPOINT_PATH)
    torch.manual_seed(42)
    images = torch.rand(3, 3, 256, 256)
    action = policy.step(images)
    assert torch.isfinite(action).all()


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_frozen():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    policy = VQBeTPolicy.from_checkpoint(CHECKPOINT_PATH)
    for name, p in policy.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"
