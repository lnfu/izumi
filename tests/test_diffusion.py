"""Tests for Diffusion Policy.

Unit tests use random weights (no network, no checkpoint).
Checkpoint tests require IZUMI_CHECKPOINT pointing to a diffusion checkpoint.
"""

import os

import pytest
import torch

from izumi.models.diffusion import (
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    TransformerForDiffusion,
)
from izumi.models.diffusion_policy import DiffusionPolicy, _DiffusionPolicyCore

# ---------------------------------------------------------------------------
# Checkpoint availability
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("IZUMI_CHECKPOINT")
HAS_CHECKPOINT = CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH)

B, T_PRED, T_OBS, A = 2, 16, 2, 7
FEATURE_DIM = 512
GLOBAL_COND_DIM = FEATURE_DIM * T_OBS  # 1024


# ---------------------------------------------------------------------------
# ConditionalResidualBlock1D
# ---------------------------------------------------------------------------


def test_residual_block_output_shape():
    block = ConditionalResidualBlock1D(in_channels=7, out_channels=256, cond_dim=512)
    x = torch.randn(B, 7, T_PRED)
    cond = torch.randn(B, 512)
    out = block(x, cond)
    assert out.shape == (B, 256, T_PRED)


def test_residual_block_same_channels():
    block = ConditionalResidualBlock1D(in_channels=256, out_channels=256, cond_dim=512)
    x = torch.randn(B, 256, T_PRED)
    cond = torch.randn(B, 512)
    out = block(x, cond)
    assert out.shape == (B, 256, T_PRED)


# ---------------------------------------------------------------------------
# ConditionalUnet1D
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unet():
    m = ConditionalUnet1D(action_dim=A, global_cond_dim=GLOBAL_COND_DIM)
    m.eval()
    return m


def test_unet_output_shape(unet):
    sample = torch.randn(B, T_PRED, A)
    t = torch.zeros(B, dtype=torch.long)
    cond = torch.randn(B, GLOBAL_COND_DIM)
    with torch.no_grad():
        out = unet(sample, t, cond)
    assert out.shape == (B, T_PRED, A)


def test_unet_output_finite(unet):
    sample = torch.randn(B, T_PRED, A)
    t = torch.zeros(B, dtype=torch.long)
    cond = torch.randn(B, GLOBAL_COND_DIM)
    with torch.no_grad():
        out = unet(sample, t, cond)
    assert torch.isfinite(out).all()


def test_unet_scalar_timestep(unet):
    sample = torch.randn(1, T_PRED, A)
    cond = torch.randn(1, GLOBAL_COND_DIM)
    with torch.no_grad():
        out = unet(sample, 50, cond)
    assert out.shape == (1, T_PRED, A)


def test_unet_state_dict_roots(unet):
    roots = {k.split(".")[0] for k in unet.state_dict()}
    assert roots == {
        "diffusion_step_encoder",
        "down_modules",
        "mid_modules",
        "up_modules",
        "final_conv",
    }


# ---------------------------------------------------------------------------
# TransformerForDiffusion
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tfd():
    m = TransformerForDiffusion(
        action_dim=A,
        obs_dim=FEATURE_DIM,
        pred_horizon=T_PRED,
        obs_horizon=T_OBS,
    )
    m.eval()
    return m


def test_transformer_output_shape(tfd):
    sample = torch.randn(B, T_PRED, A)
    t = torch.zeros(B, dtype=torch.long)
    cond = torch.randn(B, GLOBAL_COND_DIM)
    with torch.no_grad():
        out = tfd(sample, t, cond)
    assert out.shape == (B, T_PRED, A)


def test_transformer_output_finite(tfd):
    sample = torch.randn(B, T_PRED, A)
    t = torch.zeros(B, dtype=torch.long)
    cond = torch.randn(B, GLOBAL_COND_DIM)
    with torch.no_grad():
        out = tfd(sample, t, cond)
    assert torch.isfinite(out).all()


def test_transformer_state_dict_roots(tfd):
    roots = {k.split(".")[0] for k in tfd.state_dict()}
    assert roots == {
        "_dummy_variable",
        "pos_emb",
        "cond_pos_emb",
        "mask",
        "memory_mask",
        "input_emb",
        "cond_obs_emb",
        "encoder",
        "decoder",
        "ln_f",
        "head",
    }


# ---------------------------------------------------------------------------
# DiffusionPolicy (no encoder, mocked with random weights)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def policy_no_encoder():
    """DiffusionPolicy with a ConditionalUnet1D and FakeEncoder (no hub download)."""
    import torch.nn as nn

    unet = ConditionalUnet1D(action_dim=A, global_cond_dim=GLOBAL_COND_DIM)

    class FakeEncoder(nn.Module):
        feature_dim = FEATURE_DIM

        def forward(self, x):
            B, T, *_ = x.shape
            return torch.zeros(B, T, self.feature_dim)

    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    nn.Module.__init__(policy)
    policy.pred_horizon = T_PRED
    policy.obs_horizon = T_OBS
    policy.action_dim = A
    policy.num_inference_steps = 5  # fast for testing
    policy.encoder = FakeEncoder()
    policy._obs_adapter = nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=False)
    policy._obs_mask_token = nn.Parameter(torch.zeros(FEATURE_DIM))
    policy._diffusionpolicy = _DiffusionPolicyCore(unet)
    from diffusers import DDPMScheduler

    policy.scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    policy.eval()
    return policy


def test_policy_step_output_shape(policy_no_encoder):
    images = torch.zeros(T_OBS, 3, 256, 256)
    action = policy_no_encoder.step(images)
    assert action.shape == (A,)


def test_policy_step_output_finite(policy_no_encoder):
    images = torch.zeros(T_OBS, 3, 256, 256)
    action = policy_no_encoder.step(images)
    assert torch.isfinite(action).all()


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_loads_without_error():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a diffusion checkpoint to run")
    policy = DiffusionPolicy.from_checkpoint(CHECKPOINT_PATH)
    assert policy is not None


@pytest.mark.checkpoint
@pytest.mark.hub
def test_from_checkpoint_step_output_shape():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a diffusion checkpoint to run")
    policy = DiffusionPolicy.from_checkpoint(CHECKPOINT_PATH)
    images = torch.zeros(6, 3, 256, 256)  # (T_obs=6, C, H, W) — matches checkpoint obs_horizon
    action = policy.step(images)
    assert action.shape == (7,)
