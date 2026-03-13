"""Visual encoder wrapping a timm model for inference."""

import einops
import timm
import torch
import torch.nn as nn
from torchvision import transforms


class TimmEncoder(nn.Module):
    """Wraps a timm model to encode (B, T, C, H, W) image sequences.

    Replicates TimmSSL from robot-utility-models with inference-only transforms.
    State dict keys are ``model.*``, matching ``checkpoint["model"]`` from the
    original training checkpoints.
    """

    def __init__(self, model_name: str = "hf-hub:notmahi/dobb-e") -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self._transform = _build_test_transform(self.model.pretrained_cfg)

    @property
    def feature_dim(self) -> int:
        return self.model.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of images.

        Args:
            images: Float tensor ``(B, T, C, H, W)`` in [0, 1].

        Returns:
            Float tensor ``(B, T, feature_dim)``.
        """
        b = images.shape[0]
        flat = einops.rearrange(images, "b t c h w -> (b t) c h w")
        features = self.model(self._transform(flat))
        return einops.rearrange(features, "(b t) c -> b t c", b=b)


def _build_test_transform(pretrained_cfg) -> transforms.Compose:
    """Build the inference (test) transform matching the original TimmSSL.

    Replicates ``decord_transforms.create_transform`` with ``is_training=False``
    and ``crop_pct=0.875``.  No ``ToTensor`` — inputs are expected to be float
    tensors in [0, 1] already.
    """
    data_cfg = timm.data.resolve_data_config(pretrained_cfg)
    input_size = data_cfg["input_size"]  # (C, H, W)
    img_size = input_size[1]  # assume square
    crop_pct = data_cfg.get("crop_pct", 0.875)
    rescaled_size = int(img_size / crop_pct)
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    return transforms.Compose(
        [
            transforms.Resize(rescaled_size, antialias=False),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
