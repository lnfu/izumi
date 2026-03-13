"""Integration tests for TimmEncoder with the real dobb-e model.

These tests require either:
  - Network access to HuggingFace Hub (for the model architecture), OR
  - A checkpoint file containing ``checkpoint["model"]`` from the original
    robot-utility-models training run.

Running
-------
# Test dobb-e architecture (needs HF Hub, no checkpoint):
    uv run pytest tests/integration/test_encoder_checkpoint.py -m hub -v

# Test checkpoint compatibility (needs a .pt file):
    IZUMI_CHECKPOINT=/path/to/checkpoint.pt \
        uv run pytest tests/integration/test_encoder_checkpoint.py -m checkpoint -v

# Run everything:
    IZUMI_CHECKPOINT=/path/to/checkpoint.pt \
        uv run pytest tests/integration/test_encoder_checkpoint.py -v
"""

import os

import pytest
import torch

from izumi.models.encoder import TimmEncoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("IZUMI_CHECKPOINT")
HAS_CHECKPOINT = CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def dobbe_encoder():
    """TimmEncoder with the real dobb-e architecture (no pretrained weights)."""
    enc = TimmEncoder(model_name="hf-hub:notmahi/dobb-e")
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def loaded_encoder():
    """TimmEncoder with weights loaded from IZUMI_CHECKPOINT."""
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT=/path/to/checkpoint.pt to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    enc = TimmEncoder(model_name="hf-hub:notmahi/dobb-e")
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# Hub tests (need network, no checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.hub
def test_dobbe_output_shape(dobbe_encoder):
    images = torch.zeros(2, 3, 3, 256, 256)
    with torch.no_grad():
        out = dobbe_encoder(images)
    assert out.shape == (2, 3, dobbe_encoder.feature_dim)


@pytest.mark.hub
def test_dobbe_feature_dim(dobbe_encoder):
    # hf-hub:notmahi/dobb-e is a ResNet, expected to produce 512-dim features
    assert dobbe_encoder.feature_dim == 512


@pytest.mark.hub
def test_dobbe_state_dict_keys(dobbe_encoder):
    keys = list(dobbe_encoder.state_dict().keys())
    assert all(k.startswith("model.") for k in keys)


# ---------------------------------------------------------------------------
# Checkpoint tests (need IZUMI_CHECKPOINT)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_checkpoint_loads_without_error(loaded_encoder):
    """All checkpoint["model"] keys should load cleanly (strict=True)."""
    # If this fixture didn't raise, loading succeeded
    assert loaded_encoder is not None


@pytest.mark.checkpoint
def test_checkpoint_output_shape(loaded_encoder):
    images = torch.zeros(1, 3, 3, 256, 256)
    with torch.no_grad():
        out = loaded_encoder(images)
    assert out.shape == (1, 3, loaded_encoder.feature_dim)


@pytest.mark.checkpoint
def test_checkpoint_output_is_finite(loaded_encoder):
    """Real weights should produce finite (non-NaN, non-Inf) outputs."""
    torch.manual_seed(42)
    images = torch.rand(1, 1, 3, 256, 256)
    with torch.no_grad():
        out = loaded_encoder(images)
    assert torch.isfinite(out).all(), "Encoder output contains NaN or Inf"


@pytest.mark.checkpoint
def test_checkpoint_key_compatibility():
    """Verify every key in checkpoint["model"] is present in TimmEncoder."""
    if not HAS_CHECKPOINT:
        pytest.skip("Set IZUMI_CHECKPOINT=/path/to/checkpoint.pt to run")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["model"].keys())
    enc = TimmEncoder(model_name="hf-hub:notmahi/dobb-e")
    model_keys = set(enc.state_dict().keys())
    missing = ckpt_keys - model_keys
    extra = model_keys - ckpt_keys
    assert not missing, f"Keys in checkpoint but not in TimmEncoder: {missing}"
    assert not extra, f"Keys in TimmEncoder but not in checkpoint: {extra}"
