"""VQ-VAE (action tokenizer) ported from robot-utility-models.

Inference-only port.  Training methods (vqvae_update, configure_optimizers)
are omitted.  State dict keys are designed to match checkpoint["loss_fn"] after
stripping the ``_rvq.`` prefix used by the original VQBeTLossFn.
"""

import einops
import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack

# ---------------------------------------------------------------------------
# MLP encoder / decoder
# ---------------------------------------------------------------------------


class EncoderMLP(nn.Module):
    """Simple MLP: Linear -> ReLU -> [Linear -> ReLU] x layer_num -> Linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        hidden_dim: int = 128,
        layer_num: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x))


# ---------------------------------------------------------------------------
# Codebook (inference-only, buffer layout matches EuclideanCodebook)
# ---------------------------------------------------------------------------


class EuclideanCodebook(nn.Module):
    """Minimal codebook holding only the buffers needed for inference.

    Buffer names and shapes exactly match the original EuclideanCodebook so
    that checkpoints load with strict=True.

    Shapes (with num_codebooks=1):
        embed:        (1, codebook_size, dim)
        cluster_size: (1, codebook_size)
        embed_avg:    (1, codebook_size, dim)
        initted:      (1,)
    """

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        embed = torch.zeros(1, codebook_size, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(1, codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initted", torch.ones(1))  # True = already initted


class VectorQuantize(nn.Module):
    """Inference-only wrapper whose state dict matches the original VectorQuantize.

    project_in / project_out are Identity (dim == codebook_dim, no projection).
    The codebook is stored in ``_codebook`` (EuclideanCodebook).
    """

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self._codebook = EuclideanCodebook(dim=dim, codebook_size=codebook_size)


# ---------------------------------------------------------------------------
# Residual VQ
# ---------------------------------------------------------------------------


class ResidualVQ(nn.Module):
    """Residual Vector Quantization (inference-only).

    Stacks ``num_quantizers`` VectorQuantize layers.  Only implements
    ``get_codes_from_indices`` (used by VqVae.draw_code_forward).
    """

    def __init__(self, *, dim: int, num_quantizers: int, codebook_size: int) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self.layers = nn.ModuleList(
            [VectorQuantize(dim=dim, codebook_size=codebook_size) for _ in range(num_quantizers)]
        )

    @property
    def codebooks(self) -> torch.Tensor:
        """Shape: (num_quantizers, codebook_size, dim)."""
        codebooks = [layer._codebook.embed for layer in self.layers]
        stacked = torch.stack(codebooks, dim=0)  # (Q, 1, C, D)
        return rearrange(stacked, "q 1 c d -> q c d")  # (Q, C, D)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codewords from multi-quantizer indices.

        Args:
            indices: Long tensor ``(batch, ..., num_quantizers)``.

        Returns:
            Float tensor ``(num_quantizers, batch, ..., dim)``.
        """
        batch = indices.shape[0]
        indices, ps = pack([indices], "b * q")  # (B, N, Q)

        codebooks = repeat(self.codebooks, "q c d -> q b c d", b=batch)  # (Q, B, C, D)
        gather_indices = repeat(indices, "b n q -> q b n d", d=codebooks.shape[-1])  # (Q, B, N, D)

        mask = gather_indices == -1
        gather_indices = gather_indices.masked_fill(mask, 0)
        all_codes = codebooks.gather(2, gather_indices)  # (Q, B, N, D)
        all_codes = all_codes.masked_fill(mask, 0.0)

        (all_codes,) = unpack(all_codes, ps, "q b * d")  # (Q, B, ..., D)
        return all_codes


# ---------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------


class VqVae(nn.Module):
    """Action tokenizer (VQ-VAE with Residual VQ), inference-only.

    Encodes 7D actions into discrete codes; decodes codes back to actions.

    Default constructor arguments match the door_opening / VQ-BeT checkpoint:
        input_dim_h=1, input_dim_w=7, n_latent_dims=512,
        vqvae_n_embed=16, vqvae_groups=2, act_scale=1.0

    Checkpoint loading
    ------------------
    When loading from the original checkpoint's ``loss_fn`` state dict, the
    VqVae is stored under the ``_rvq.`` key prefix.  Use
    ``VqVae.from_loss_fn_state_dict()`` to handle the prefix stripping.
    """

    def __init__(
        self,
        input_dim_h: int = 1,
        input_dim_w: int = 7,
        n_latent_dims: int = 512,
        vqvae_n_embed: int = 16,
        vqvae_groups: int = 2,
        act_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.act_scale = act_scale

        enc_in = input_dim_w if input_dim_h == 1 else input_dim_w * input_dim_h
        self.encoder = EncoderMLP(input_dim=enc_in, output_dim=n_latent_dims)
        self.decoder = EncoderMLP(input_dim=n_latent_dims, output_dim=enc_in)
        self.vq_layer = ResidualVQ(
            dim=n_latent_dims,
            num_quantizers=vqvae_groups,
            codebook_size=vqvae_n_embed,
        )
        self.requires_grad_(False)

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def draw_code_forward(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete code indices into summed latent vectors.

        Args:
            encoding_indices: Long tensor ``(NT, num_quantizers)`` — code
                indices for each quantizer layer.

        Returns:
            Float tensor ``(NT, 1, n_latent_dims)`` — summed codeword embeddings.
        """
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            return z_embed.sum(dim=0)

    def get_action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector into an action sequence.

        Args:
            latent: Float tensor ``(NT, n_latent_dims)``.

        Returns:
            Float tensor ``(NT, input_dim_h, input_dim_w)``.
        """
        output = self.decoder(latent) * self.act_scale
        return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_loss_fn_state_dict(
        cls,
        loss_fn_state_dict: dict,
        **kwargs,
    ) -> "VqVae":
        """Build and load a VqVae from the original ``checkpoint["loss_fn"]``.

        Strips the ``_rvq.`` prefix used by VQBeTLossFn before calling
        ``load_state_dict``.
        """
        prefix = "_rvq."
        vqvae_sd = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in loss_fn_state_dict.items()
            if k.startswith(prefix)
        }
        model = cls(**kwargs)
        model.load_state_dict(vqvae_sd, strict=True)
        return model
