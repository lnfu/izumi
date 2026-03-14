"""Tests for VqVae (action tokenizer).

Unit tests use random weights.  Checkpoint tests require IZUMI_CHECKPOINT.
"""

import os

import pytest
import torch

from izumi.models.vqvae import EncoderMLP, ResidualVQ, VqVae

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("IZUMI_CHECKPOINT")
HAS_CHECKPOINT = CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def vqvae():
    m = VqVae()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Unit tests (no checkpoint needed)
# ---------------------------------------------------------------------------


def test_encoder_mlp_shape():
    mlp = EncoderMLP(input_dim=7, output_dim=512)
    x = torch.randn(4, 7)
    assert mlp(x).shape == (4, 512)


def test_residual_vq_codebooks_shape():
    rvq = ResidualVQ(dim=512, num_quantizers=2, codebook_size=16)
    cb = rvq.codebooks
    assert cb.shape == (2, 16, 512)


def test_get_codes_from_indices_shape():
    rvq = ResidualVQ(dim=512, num_quantizers=2, codebook_size=16)
    # Randomly initialise codebook entries
    for layer in rvq.layers:
        nn.init.normal_(layer._codebook.embed)
    indices = torch.randint(0, 16, (8, 2))  # (NT, G)
    codes = rvq.get_codes_from_indices(indices)  # (G, NT, n, dim)
    assert codes.shape[0] == 2  # num_quantizers
    assert codes.shape[1] == 8  # batch / NT


def test_draw_code_forward_shape(vqvae):
    # draw_code_forward returns (NT, dim); caller does .view(NT, -1, D)
    indices = torch.randint(0, 16, (8, 2))  # (NT, G=2)
    out = vqvae.draw_code_forward(indices)
    assert out.shape == (8, 512)


def test_get_action_from_latent_shape(vqvae):
    latent = torch.randn(8, 512)
    actions = vqvae.get_action_from_latent(latent)
    assert actions.shape == (8, 1, 7)  # (NT, input_dim_h=1, input_dim_w=7)


def test_round_trip_shapes(vqvae):
    """draw_code_forward -> view -> get_action_from_latent mirrors vqbet.py usage."""
    NT, G, D = 6, 2, 512
    indices = torch.randint(0, 16, (NT, G))
    centers = vqvae.draw_code_forward(indices).view(NT, -1, D)  # (NT, 1, D)
    latent = einops_rearrange(centers, "NT G D -> NT (G D)")  # (NT, D)
    actions = vqvae.get_action_from_latent(latent)
    assert actions.shape == (NT, 1, 7)


def test_no_grad_params(vqvae):
    """VqVae should have requires_grad=False on all parameters (frozen)."""
    for name, p in vqvae.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"


def test_state_dict_key_prefixes(vqvae):
    """State dict must have encoder/decoder/vq_layer keys (no unexpected roots)."""
    keys = list(vqvae.state_dict().keys())
    roots = {k.split(".")[0] for k in keys}
    assert roots == {"encoder", "decoder", "vq_layer"}


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_checkpoint_state_dict_keys():
    """All _rvq.* keys in loss_fn must load cleanly into VqVae (strict=True)."""
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vqvae = VqVae.from_loss_fn_state_dict(ckpt["loss_fn"])
    assert vqvae is not None


@pytest.mark.checkpoint
def test_checkpoint_draw_code_forward():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vqvae = VqVae.from_loss_fn_state_dict(ckpt["loss_fn"])
    indices = torch.randint(0, 16, (4, 2))
    out = vqvae.draw_code_forward(indices)
    assert out.shape == (4, 512)
    assert torch.isfinite(out).all()


@pytest.mark.checkpoint
def test_checkpoint_get_action_from_latent():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vqvae = VqVae.from_loss_fn_state_dict(ckpt["loss_fn"])
    latent = torch.randn(4, 512)
    actions = vqvae.get_action_from_latent(latent)
    assert actions.shape == (4, 1, 7)
    assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# Helper (avoid import at module level for test isolation)
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402  (used in test_get_codes_from_indices_shape)


def einops_rearrange(tensor, pattern):
    import einops

    return einops.rearrange(tensor, pattern)
