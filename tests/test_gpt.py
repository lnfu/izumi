"""Tests for GPT (nanoGPT-style transformer backbone).

Unit tests use random weights.  Checkpoint tests require IZUMI_CHECKPOINT
pointing to a VQ-BeT checkpoint (not diffusion).
"""

import os

import pytest
import torch

from izumi.models.gpt import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("IZUMI_CHECKPOINT")
HAS_CHECKPOINT = CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def cfg():
    return GPTConfig()


@pytest.fixture(scope="module")
def gpt(cfg):
    m = GPT(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Unit tests (no checkpoint needed)
# ---------------------------------------------------------------------------


def test_output_shape(gpt, cfg):
    x = torch.randn(2, 3, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert out.shape == (2, 3, cfg.output_dim)


def test_output_shape_single_step(gpt, cfg):
    x = torch.randn(1, 1, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert out.shape == (1, 1, cfg.output_dim)


def test_output_shape_full_context(gpt, cfg):
    x = torch.randn(4, cfg.block_size, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert out.shape == (4, cfg.block_size, cfg.output_dim)


def test_causal_mask(cfg):
    """Output at position t must not depend on positions > t."""
    gpt = GPT(cfg)
    gpt.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 5, cfg.input_dim)

    with torch.no_grad():
        out_full = gpt(x)
        out_prefix = gpt(x[:, :3])

    # First 3 outputs must match when computed from prefix vs full sequence
    assert torch.allclose(out_full[:, :3], out_prefix, atol=1e-5), (
        "GPT is not causal: prefix outputs differ"
    )


def test_output_is_finite(gpt, cfg):
    x = torch.randn(2, 3, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert torch.isfinite(out).all()


def test_state_dict_key_roots(gpt):
    keys = list(gpt.state_dict().keys())
    roots = {k.split(".")[0] for k in keys}
    assert roots == {"transformer", "lm_head"}


def test_causal_mask_buffer_present(gpt, cfg):
    """Each block must have a registered causal mask buffer."""
    for i in range(cfg.n_layer):
        assert hasattr(gpt.transformer.h[i].attn, "bias"), f"h.{i}.attn missing bias buffer"
        buf = gpt.transformer.h[i].attn.bias
        assert buf.shape == (1, 1, cfg.block_size, cfg.block_size)


def test_block_count(gpt, cfg):
    assert len(gpt.transformer.h) == cfg.n_layer


def test_sequence_too_long_raises(gpt, cfg):
    x = torch.randn(1, cfg.block_size + 1, cfg.input_dim)
    with pytest.raises(AssertionError):
        gpt(x)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_checkpoint_loads_without_error():
    """_vqbet._gpt.* keys in loss_fn must load cleanly into GPT (strict=True)."""
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    gpt = GPT.from_loss_fn_state_dict(ckpt["loss_fn"])
    assert gpt is not None


@pytest.mark.checkpoint
def test_checkpoint_output_shape():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    gpt = GPT.from_loss_fn_state_dict(ckpt["loss_fn"])
    gpt.eval()
    cfg = gpt.config
    x = torch.randn(1, 3, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert out.shape == (1, 3, cfg.output_dim)


@pytest.mark.checkpoint
def test_checkpoint_output_is_finite():
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT to a VQ-BeT checkpoint to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    gpt = GPT.from_loss_fn_state_dict(ckpt["loss_fn"])
    gpt.eval()
    cfg = gpt.config
    torch.manual_seed(42)
    x = torch.randn(1, 3, cfg.input_dim)
    with torch.no_grad():
        out = gpt(x)
    assert torch.isfinite(out).all()
