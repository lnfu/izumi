"""GPT (nanoGPT-style transformer backbone) ported from robot-utility-models.

Inference-only port.  State dict key layout matches the original GPT inside
VQBehaviorTransformer so ``checkpoint["loss_fn"]`` weights load with
strict=True after stripping the ``_vqbet._gpt_model.`` prefix.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 50
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 120
    dropout: float = 0.1
    bias: bool = True  # bias in Linear and LayerNorm layers


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (head_size**-0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    """Inner transformer body — exists solely to produce the correct key prefix
    (``transformer.*``) in the state dict."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.wte(x) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------


class GPT(nn.Module):
    """nanoGPT-style transformer backbone (inference-only).

    Takes continuous observation features ``(B, T, input_dim)`` and produces
    contextualised embeddings ``(B, T, output_dim)``.

    State dict key layout (matches checkpoint ``_vqbet._gpt_model.*``):
        transformer.wte.{weight,bias}
        transformer.wpe.weight
        transformer.h.{i}.ln_1.{weight,bias}
        transformer.h.{i}.attn.bias          (causal mask buffer)
        transformer.h.{i}.attn.c_attn.{weight,bias}
        transformer.h.{i}.attn.c_proj.{weight,bias}
        transformer.h.{i}.ln_2.{weight,bias}
        transformer.h.{i}.mlp.c_fc.{weight,bias}
        transformer.h.{i}.mlp.c_proj.{weight,bias}
        transformer.ln_f.{weight,bias}
        lm_head.weight

    Checkpoint loading
    ------------------
    Inside the original checkpoint the GPT lives under
    ``_vqbet._gpt_model.*`` in ``loss_fn``.  Use
    ``GPT.from_loss_fn_state_dict()`` to handle the prefix stripping.
    """

    def __init__(self, config: GPTConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GPTConfig()
        self.config = config

        self.transformer = _Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor ``(B, T, input_dim)``

        Returns:
            Float tensor ``(B, T, output_dim)``
        """
        B, T, _ = x.size()
        assert self.config.block_size >= T, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )
        return self.lm_head(self.transformer(x))

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_loss_fn_state_dict(
        cls,
        loss_fn_state_dict: dict,
        config: GPTConfig | None = None,
    ) -> "GPT":
        """Build and load a GPT from the original ``checkpoint["loss_fn"]``.

        Strips the ``_vqbet._gpt_model.`` prefix used by VQBeTLossFn.
        """
        prefix = "_vqbet._gpt_model."
        gpt_sd = {
            k[len(prefix) :]: v for k, v in loss_fn_state_dict.items() if k.startswith(prefix)
        }
        model = cls(config)
        model.load_state_dict(gpt_sd, strict=True)
        return model
