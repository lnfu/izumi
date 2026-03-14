"""VQ-BeT policy ported from robot-utility-models.

Inference-only port.  Combines GPT backbone, residual VQ-VAE action tokenizer,
and two-stage code selection heads into a single policy module.

Checkpoint key layout inside ``checkpoint["loss_fn"]``:
    _obs_adapter.weight
    _obs_mask_token
    _vqbet._gpt_model.*
    _vqbet._vqvae_model.*
    _vqbet._map_to_cbet_preds_bin1.*
    _vqbet._map_to_cbet_preds_bin2.*
    _vqbet._map_to_cbet_preds_offset.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from izumi.models.encoder import TimmEncoder
from izumi.models.gpt import GPT, GPTConfig
from izumi.models.vqvae import VqVae


def _mlp(dims: list[int], dropout: float = 0.1) -> nn.Sequential:
    """Build Linear-ReLU-Dropout MLP matching Sequential index layout of checkpoints.

    Index layout: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, ...
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VQ-BeT core
# ---------------------------------------------------------------------------


class VQBehaviorTransformer(nn.Module):
    """GPT + VQ-VAE + two-stage code selection heads (inference-only).

    State dict roots match ``_vqbet.*`` inside ``loss_fn``:
        _gpt_model.*
        _vqvae_model.*
        _map_to_cbet_preds_bin1.*
        _map_to_cbet_preds_bin2.*
        _map_to_cbet_preds_offset.*
    """

    def __init__(
        self,
        gpt_config: GPTConfig | None = None,
        vqvae_groups: int = 2,
        cbet_num_bins: int = 16,
        bin_hidden_dim: int = 512,
        offset_hidden_dim: int = 1024,
        n_action: int = 7,
        input_dim_h: int = 1,
    ) -> None:
        super().__init__()
        if gpt_config is None:
            gpt_config = GPTConfig()

        self.cbet_num_bins = cbet_num_bins
        self.n_groups = vqvae_groups
        self.input_dim_h = input_dim_h
        self.n_action = n_action

        obs_dim = gpt_config.output_dim  # 256

        self._gpt_model = GPT(gpt_config)
        self._vqvae_model = VqVae(
            vqvae_n_embed=cbet_num_bins,
            vqvae_groups=vqvae_groups,
            input_dim_h=input_dim_h,
            input_dim_w=n_action,
        )
        self._map_to_cbet_preds_bin1 = _mlp(
            [obs_dim, bin_hidden_dim, bin_hidden_dim, cbet_num_bins]
        )
        self._map_to_cbet_preds_bin2 = _mlp(
            [obs_dim + cbet_num_bins, bin_hidden_dim, cbet_num_bins]
        )
        self._map_to_cbet_preds_offset = _mlp(
            [
                obs_dim,
                offset_hidden_dim,
                offset_hidden_dim,
                vqvae_groups * cbet_num_bins * input_dim_h * n_action,
            ]
        )

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """Run one inference step.

        Args:
            obs: Float tensor ``(T, obs_dim)`` — adapted observation sequence.

        Returns:
            Float tensor ``(T, input_dim_h, n_action)`` — predicted actions.
        """
        T = obs.shape[0]
        gpt_out = self._gpt_model(obs.unsqueeze(0)).squeeze(0)  # (T, obs_dim)

        # Stage 1: first code
        bin1_logits = self._map_to_cbet_preds_bin1(gpt_out)  # (T, C)
        code_1 = bin1_logits.argmax(dim=-1)  # (T,)

        # Stage 2: second code conditioned on first
        code_1_onehot = F.one_hot(code_1, self.cbet_num_bins).float()  # (T, C)
        bin2_logits = self._map_to_cbet_preds_bin2(
            torch.cat([gpt_out, code_1_onehot], dim=-1)
        )  # (T, C)
        code_2 = bin2_logits.argmax(dim=-1)  # (T,)

        # Decode action from codebook
        indices = torch.stack([code_1, code_2], dim=-1)  # (T, G=2)
        drawn = self._vqvae_model.draw_code_forward(indices)  # (T, n_latent_dims)
        actions = self._vqvae_model.get_action_from_latent(drawn)  # (T, W, A)

        # Offset residual
        G, C, W, A = self.n_groups, self.cbet_num_bins, self.input_dim_h, self.n_action
        offset = self._map_to_cbet_preds_offset(gpt_out)  # (T, G*C*W*A)
        offset = offset.view(T, G, C, W, A)

        # Gather offset at sampled code indices for each quantizer
        codes = torch.stack([code_1, code_2], dim=1)  # (T, G)
        idx = codes.view(T, G, 1, 1, 1).expand(T, G, 1, W, A)
        sampled_offset = offset.gather(2, idx).squeeze(2)  # (T, G, W, A)
        sampled_offset = sampled_offset.sum(dim=1)  # (T, W, A)

        return actions + sampled_offset  # (T, W, A)


# ---------------------------------------------------------------------------
# Full policy
# ---------------------------------------------------------------------------


class VQBeTPolicy(nn.Module):
    """End-to-end VQ-BeT policy: visual encoder + obs adapter + VQBehaviorTransformer.

    Inference-only.  Use ``VQBeTPolicy.from_checkpoint(path)`` to load weights.

    The encoder is loaded from ``checkpoint["model"]``; everything else from
    ``checkpoint["loss_fn"]``.
    """

    def __init__(
        self,
        encoder_model_name: str = "hf-hub:notmahi/dobb-e",
        obs_dim: int = 256,
        **vqbet_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = TimmEncoder(model_name=encoder_model_name)
        feature_dim = self.encoder.feature_dim  # 512 for dobb-e

        # Matches loss_fn keys directly (no prefix stripping needed)
        self._obs_adapter = nn.Linear(feature_dim, obs_dim, bias=False)
        self._obs_mask_token = nn.Parameter(torch.zeros(obs_dim))
        self._vqbet = VQBehaviorTransformer(**vqbet_kwargs)

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        """Run one inference step.

        Args:
            images: Float tensor ``(T, C, H, W)`` or ``(1, T, C, H, W)``.
                    Values in ``[0, 1]``, shape ``(3, 256, 256)`` per frame.

        Returns:
            Float tensor ``(n_action,)`` — action for the current timestep.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)  # (1, T, C, H, W)
        enc = self.encoder(images)  # (1, T, feature_dim)
        obs = self._obs_adapter(enc).squeeze(0)  # (T, obs_dim)
        a_hat = self._vqbet.step(obs)  # (T, W, A)
        return a_hat[-1, -1, :]  # (A,) — last timestep

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "VQBeTPolicy":
        """Load a VQBeTPolicy from an original robot-utility-models checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            device: Device to load tensors onto.

        Returns:
            Fully loaded ``VQBeTPolicy`` in eval mode with frozen parameters.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        policy = cls(**kwargs)

        # Encoder lives in checkpoint["model"]
        policy.encoder.load_state_dict(ckpt["model"])

        loss_fn_sd = ckpt["loss_fn"]

        # obs_adapter (weight-only linear)
        policy._obs_adapter.load_state_dict({"weight": loss_fn_sd["_obs_adapter.weight"]})

        # obs_mask_token (learnable parameter)
        policy._obs_mask_token.data.copy_(loss_fn_sd["_obs_mask_token"])

        # VQBehaviorTransformer
        prefix = "_vqbet."
        vqbet_sd = {k[len(prefix) :]: v for k, v in loss_fn_sd.items() if k.startswith(prefix)}
        policy._vqbet.load_state_dict(vqbet_sd, strict=True)

        policy.eval()
        policy.requires_grad_(False)
        return policy.to(device)
