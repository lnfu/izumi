"""Denoising network architectures for Diffusion Policy.

Ported from robot-utility-models / chi et al. diffusion_policy codebase.
Inference-only.  Two variants:
  - ConditionalUnet1D  — CNN with FiLM conditioning (primary)
  - TransformerForDiffusion — encoder-decoder transformer (alternative)

Both accept:
    sample    (B, T_pred, action_dim)   noisy action sequence
    timestep  (B,) or scalar            diffusion timestep
    global_cond (B, cond_dim)           flattened obs + time conditioning
and return a noise prediction of the same shape as ``sample``.
"""

import itertools
import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position / timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# ConditionalUnet1D building blocks
# ---------------------------------------------------------------------------


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1d residual blocks with FiLM conditioning.

    State dict layout::
        blocks.0.{0,1,2}  Conv1d / GroupNorm / Mish
        blocks.1.{0,1,2}  Conv1d / GroupNorm / Mish
        cond_encoder.{0,1,2}  Mish / Linear / Unflatten
        residual_conv  Conv1d or Identity
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
                    nn.GroupNorm(n_groups, out_channels),
                    nn.Mish(),
                ),
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
                    nn.GroupNorm(n_groups, out_channels),
                    nn.Mish(),
                ),
            ]
        )
        # FiLM: produces (scale, bias) from conditioning vector
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)  # (B, out_channels*2, 1)
        scale, bias = embed.chunk(2, dim=1)
        out = out * (scale + 1) + bias  # FiLM
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# ConditionalUnet1D
# ---------------------------------------------------------------------------


class ConditionalUnet1D(nn.Module):
    """1-D U-Net with global FiLM conditioning for diffusion denoising.

    Architecture (default down_dims=[256, 512, 1024]):
        - 3 down levels: each has 2 ResBlocks + optional downsample
        - 2 mid ResBlocks at bottleneck
        - 3 up levels: each has 2 ResBlocks + optional upsample + skip
        - Final ResBlock + Conv1d(→action_dim)

    State dict roots::
        diffusion_step_encoder.*
        down_modules.*
        mid_modules.*
        up_modules.*
        final_conv.*
    """

    def __init__(
        self,
        action_dim: int = 7,
        global_cond_dim: int = 1024,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()

        # Timestep embedding: sinusoidal → MLP
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        all_dims = [action_dim, *down_dims]
        in_out = list(itertools.pairwise(all_dims))

        # Down path
        self.down_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        nn.Identity()
                        if is_last
                        else nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1),
                    ]
                )
            )

        # Bottleneck
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        # Up path (skip connections double the channels).
        # Always upsample at every level so sequence length is preserved:
        # down path has len(in_out)-1 downsamples; we do len(in_out)-1 upsamples here.
        self.up_modules = nn.ModuleList()
        for _i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, stride=2, padding=1),
                    ]
                )
            )

        # Final projection
        self.final_conv = nn.ModuleDict(
            {
                "resnet": ConditionalResidualBlock1D(
                    down_dims[0], down_dims[0], cond_dim, kernel_size, n_groups
                ),
                "conv": nn.Conv1d(down_dims[0], action_dim, 1),
            }
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample:      (B, T_pred, action_dim)
            timestep:    (B,) or scalar int/long
            global_cond: (B, global_cond_dim)

        Returns:
            (B, T_pred, action_dim) — predicted noise
        """
        # Broadcast scalar timestep to batch
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])

        # (B, T, A) → (B, A, T) for Conv1d
        x = sample.moveaxis(-1, -2)

        t_emb = self.diffusion_step_encoder(timestep)  # (B, embed_dim)
        g = torch.cat([t_emb, global_cond], dim=-1)  # (B, cond_dim)

        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, g)
            x = resnet2(x, g)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, g)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet(x, g)
            x = resnet2(x, g)
            x = upsample(x)

        x = self.final_conv["resnet"](x, g)
        x = self.final_conv["conv"](x)

        return x.moveaxis(-2, -1)  # (B, T, A)


# ---------------------------------------------------------------------------
# TransformerForDiffusion
# ---------------------------------------------------------------------------


class TransformerForDiffusion(nn.Module):
    """Transformer denoiser matching robot-utility-models checkpoint layout.

    Timestep is processed by a sinusoidal embedding followed by a 2-layer MLP
    (``encoder``).  The time token is prepended to the projected observation
    tokens to form the cross-attention memory.  Noisy action tokens are decoded
    via ``decoder`` and projected to action space.

    State dict roots::
        _dummy_variable
        pos_emb
        cond_pos_emb
        mask
        memory_mask
        input_emb.*
        cond_obs_emb.*
        encoder.*    (time MLP: Linear → GELU → Linear, indices 0 / 2)
        decoder.*
        ln_f.*
        head.*
    """

    def __init__(
        self,
        action_dim: int = 7,
        obs_dim: int = 512,
        pred_horizon: int = 11,
        obs_horizon: int = 6,
        d_model: int = 768,
        nhead: int = 8,
        num_decoder_layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        T_cond = obs_horizon + 1  # +1 for the time token

        # Sinusoidal timestep embedding (no learnable params)
        self._time_emb = SinusoidalPosEmb(d_model)

        # Time MLP — indices 0/2 match checkpoint keys encoder.0.* / encoder.2.*
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.cond_obs_emb = nn.Linear(obs_dim, d_model)
        self.input_emb = nn.Linear(action_dim, d_model)

        # Positional embeddings (initialised to zeros, learned via checkpoint)
        self.register_buffer("pos_emb", torch.zeros(1, pred_horizon, d_model))
        self.register_buffer("cond_pos_emb", torch.zeros(1, T_cond, d_model))

        # Causal self-attention mask: 0 = attended, -inf = masked
        causal = torch.zeros(pred_horizon, pred_horizon)
        causal.fill_(float("-inf"))
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("mask", causal)

        # Cross-attention mask: attend to all condition tokens
        self.register_buffer("memory_mask", torch.zeros(pred_horizon, T_cond))

        # EMA dummy variable (present in checkpoint, required for strict loading)
        self.register_buffer("_dummy_variable", torch.zeros(0))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample:      (B, T_pred, action_dim)
            timestep:    (B,) or scalar
            global_cond: (B, T_obs * obs_dim)  flattened observation features

        Returns:
            (B, T_pred, action_dim) — predicted noise
        """
        B, _T_pred, _ = sample.shape
        T_cond = self.cond_pos_emb.shape[1]
        obs_horizon = T_cond - 1

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(B)

        # Time token: sinusoidal → MLP
        t_emb = self._time_emb(timestep)  # (B, d_model)
        t_tok = self.encoder(t_emb).unsqueeze(1)  # (B, 1, d_model)

        # Obs tokens
        obs_dim = global_cond.shape[-1] // obs_horizon
        obs = global_cond.view(B, obs_horizon, obs_dim)  # (B, T_obs, obs_dim)
        obs_tok = self.cond_obs_emb(obs)  # (B, T_obs, d_model)

        # Condition memory: [time_token, obs_tokens] + positional
        memory = torch.cat([t_tok, obs_tok], dim=1) + self.cond_pos_emb  # (B, T_cond, d_model)

        # Query: noisy action tokens + positional
        query = self.input_emb(sample) + self.pos_emb  # (B, T_pred, d_model)

        out = self.decoder(
            query,
            memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask,
        )
        out = self.ln_f(out)
        return self.head(out)  # (B, T_pred, action_dim)
