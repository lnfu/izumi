"""Diffusion Policy ported from robot-utility-models.

Inference-only.  Combines:
  - TimmEncoder (visual encoder)
  - _DiffusionPolicyCore holding both noise_pred_net and ema_noise_pred_net
    (TransformerForDiffusion by default)
  - DDPMScheduler (100 steps, squaredcos_cap_v2, epsilon prediction)

Checkpoint key layout inside ``checkpoint["loss_fn"]``::
    _obs_adapter.weight            Linear(feature_dim, feature_dim, bias=False)
    _obs_mask_token                Parameter (feature_dim,)
    _diffusionpolicy.noise_pred_net.*
    _diffusionpolicy.ema_noise_pred_net.*
"""

import copy

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from izumi.models.diffusion import TransformerForDiffusion
from izumi.models.encoder import TimmEncoder


class _DiffusionPolicyCore(nn.Module):
    """Holds both the live and EMA copies of the denoising network.

    State dict roots match ``_diffusionpolicy.*`` inside ``loss_fn``::
        noise_pred_net.*
        ema_noise_pred_net.*
    """

    def __init__(self, noise_pred_net: nn.Module) -> None:
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.ema_noise_pred_net = copy.deepcopy(noise_pred_net)


class DiffusionPolicy(nn.Module):
    """Visual encoder + denoising network + DDPM scheduler (inference-only).

    Use ``DiffusionPolicy.from_checkpoint(path)`` to load a trained checkpoint.
    """

    def __init__(
        self,
        encoder_model_name: str = "hf-hub:notmahi/dobb-e",
        noise_pred_net: nn.Module | None = None,
        pred_horizon: int = 11,
        obs_horizon: int = 6,
        action_dim: int = 7,
        num_inference_steps: int = 100,
    ) -> None:
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps

        self.encoder = TimmEncoder(model_name=encoder_model_name)
        feature_dim = self.encoder.feature_dim  # 512

        self._obs_adapter = nn.Linear(feature_dim, feature_dim, bias=False)
        self._obs_mask_token = nn.Parameter(torch.zeros(feature_dim))

        if noise_pred_net is None:
            noise_pred_net = TransformerForDiffusion(
                action_dim=action_dim,
                obs_dim=feature_dim,
                pred_horizon=pred_horizon,
                obs_horizon=obs_horizon,
            )
        self._diffusionpolicy = _DiffusionPolicyCore(noise_pred_net)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )

    def _encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images → adapted features → flat global condition vector.

        Args:
            images: ``(1, T_obs, C, H, W)`` or ``(T_obs, C, H, W)``

        Returns:
            ``(1, T_obs * feature_dim)``
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        feats = self.encoder(images)  # (1, T_obs, feature_dim)
        adapted = self._obs_adapter(feats)  # (1, T_obs, feature_dim)
        return adapted.flatten(start_dim=1)  # (1, T_obs * feature_dim)

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        """Run a full denoising rollout from a visual observation.

        Args:
            images: ``(T_obs, C, H, W)`` or ``(1, T_obs, C, H, W)``.
                    Values in ``[0, 1]``.

        Returns:
            ``(action_dim,)`` — first action of the denoised trajectory.
        """
        device = next(self.parameters()).device
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device)

        global_cond = self._encode_obs(images)  # (1, T_obs * feature_dim)

        # Start from Gaussian noise
        noisy = torch.randn(1, self.pred_horizon, self.action_dim, device=device)

        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in self.scheduler.timesteps:
            noise_pred = self._diffusionpolicy.ema_noise_pred_net(noisy, t, global_cond)
            noisy = self.scheduler.step(noise_pred, t, noisy).prev_sample

        return noisy[0, 0, :]  # first predicted action

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str | torch.device = "cpu",
        noise_pred_net: nn.Module | None = None,
        **kwargs,
    ) -> "DiffusionPolicy":
        """Load a DiffusionPolicy from an original robot-utility-models checkpoint.

        The denoising networks are stored under ``checkpoint["loss_fn"]``
        with keys prefixed by ``_diffusionpolicy.``.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            device: Device to load onto.
            noise_pred_net: Pre-constructed denoising network whose architecture
                matches the checkpoint.  If ``None``, a default
                ``TransformerForDiffusion`` is used.

        Returns:
            Fully loaded ``DiffusionPolicy`` in eval mode with frozen parameters.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        policy = cls(noise_pred_net=noise_pred_net, **kwargs)

        # Encoder
        policy.encoder.load_state_dict(ckpt["model"])

        loss_fn = ckpt["loss_fn"]

        # Obs adapter and mask token
        policy._obs_adapter.load_state_dict({"weight": loss_fn["_obs_adapter.weight"]})
        policy._obs_mask_token.data.copy_(loss_fn["_obs_mask_token"])

        # Core denoising networks (_diffusionpolicy.noise_pred_net.* and ema_*)
        prefix = "_diffusionpolicy."
        dp_sd = {k[len(prefix) :]: v for k, v in loss_fn.items() if k.startswith(prefix)}
        policy._diffusionpolicy.load_state_dict(dp_sd, strict=True)

        policy.eval()
        policy.requires_grad_(False)
        return policy.to(device)
