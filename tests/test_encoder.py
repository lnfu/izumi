"""Tests for TimmEncoder.

Uses resnet18 (no hub download) for unit tests.  Tests requiring the actual
dobb-e checkpoint are in tests/integration/.
"""

import pytest
import torch

from izumi.models.encoder import TimmEncoder


@pytest.fixture(scope="module")
def encoder():
    enc = TimmEncoder(model_name="resnet18")
    enc.eval()
    return enc


def test_output_shape(encoder):
    images = torch.zeros(2, 3, 3, 256, 256)
    with torch.no_grad():
        out = encoder(images)
    assert out.shape == (2, 3, encoder.feature_dim)


def test_feature_dim(encoder):
    # resnet18 outputs 512-dim features
    assert encoder.feature_dim == 512


def test_state_dict_keys_are_model_prefixed(encoder):
    """All keys must start with 'model.' to match checkpoint["model"] format."""
    keys = list(encoder.state_dict().keys())
    assert len(keys) > 0
    non_model_keys = [k for k in keys if not k.startswith("model.")]
    assert non_model_keys == [], f"Unexpected state dict keys: {non_model_keys}"


def test_deterministic_in_eval(encoder):
    torch.manual_seed(0)
    images = torch.rand(1, 2, 3, 256, 256)
    with torch.no_grad():
        out1 = encoder(images)
        out2 = encoder(images)
    assert torch.allclose(out1, out2)


def test_batch_independence(encoder):
    """Encoding images one-by-one should match batch encoding."""
    torch.manual_seed(1)
    images = torch.rand(2, 1, 3, 256, 256)
    with torch.no_grad():
        batched = encoder(images)
        single0 = encoder(images[0:1])
        single1 = encoder(images[1:2])
    assert torch.allclose(batched[0], single0[0], atol=1e-5)
    assert torch.allclose(batched[1], single1[0], atol=1e-5)


def test_load_state_dict(encoder):
    """State dict round-trip preserves weights."""
    sd = encoder.state_dict()
    enc2 = TimmEncoder(model_name="resnet18")
    enc2.load_state_dict(sd)
    images = torch.zeros(1, 1, 3, 224, 224)
    with torch.no_grad():
        out1 = encoder(images)
        out2 = enc2.eval()(images)
    assert torch.allclose(out1, out2)
