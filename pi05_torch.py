"""Pure-PyTorch reimplementation of PI05 (pi05_libero).

This file is self-contained: no HuggingFace `transformers`, no JAX, no
monkey-patches. It mirrors the inference flow of `src/openpi/models/pi0.py`
and `src/openpi/models_pytorch/pi0_pytorch.py`, structured to match the
diagram in `model.md`:

    3 RGB images -> SigLIP ViT-So400M/14 -> per-image [B, 256, 2048]
    text prompt   -> token embedder       -> [B, L, 2048]
                  prefix tokens [B, 3*256+L, 2048]
    noisy actions [B, 10, 32] + sincos(time) -> action expert [B, 10, 1024]
    18 x two-expert Gemma blocks (with adaRMS on the action side)
    last 10 tokens -> action_out_proj -> v_t [B, 10, 32]
    Euler integrate from t=1 to t=0 over `num_steps` -> action chunk

Random initialization only. Run as:

    python pi05_torch.py
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SiglipConfig:
    image_size: int = 224
    patch_size: int = 14
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    projection_dim: int = 2048


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclasses.dataclass(frozen=True)
class Pi05Config:
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    vocab_size: int = 257_152
    siglip: SiglipConfig = dataclasses.field(default_factory=SiglipConfig)
    paligemma: GemmaConfig = dataclasses.field(
        default_factory=lambda: GemmaConfig(
            width=2048, depth=18, mlp_dim=16_384,
            num_heads=8, num_kv_heads=1, head_dim=256,
        )
    )
    action_expert: GemmaConfig = dataclasses.field(
        default_factory=lambda: GemmaConfig(
            width=1024, depth=18, mlp_dim=4096,
            num_heads=8, num_kv_heads=1, head_dim=256,
        )
    )


# ---------------------------------------------------------------------------
# SigLIP (ViT-So400M/14) image encoder
# ---------------------------------------------------------------------------


class SiglipMLP(nn.Module):
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipAttention(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        assert width % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.out_proj(out)


class SiglipBlock(nn.Module):
    def __init__(self, cfg: SiglipConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.width, eps=1e-6)
        self.self_attn = SiglipAttention(cfg.width, cfg.num_heads)
        self.layer_norm2 = nn.LayerNorm(cfg.width, eps=1e-6)
        self.mlp = SiglipMLP(cfg.width, cfg.mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipVisionEncoder(nn.Module):
    """ViT-So400M/14 patch-embedding + 27 transformer blocks + projector."""

    def __init__(self, cfg: SiglipConfig):
        super().__init__()
        self.cfg = cfg
        num_patches = (cfg.image_size // cfg.patch_size) ** 2  # 256
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=cfg.width,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.position_embedding = nn.Embedding(num_patches, cfg.width)
        self.register_buffer(
            "position_ids", torch.arange(num_patches).unsqueeze(0), persistent=False
        )
        self.layers = nn.ModuleList([SiglipBlock(cfg) for _ in range(cfg.depth)])
        self.post_layernorm = nn.LayerNorm(cfg.width, eps=1e-6)
        # Multi-modal projector: 1152 -> 2048
        self.multi_modal_projector = nn.Linear(cfg.width, cfg.projection_dim)

    def forward(self, image_bchw: Tensor) -> Tensor:
        """image_bchw: float [B, 3, 224, 224] in [-1, 1]. Returns [B, 256, 2048]."""
        x = self.patch_embedding(image_bchw)            # [B, 1152, 16, 16]
        x = x.flatten(2).transpose(1, 2)                # [B, 256, 1152]
        x = x + self.position_embedding(self.position_ids)
        for block in self.layers:
            x = block(x)
        x = self.post_layernorm(x)
        return self.multi_modal_projector(x)            # [B, 256, 2048]


# ---------------------------------------------------------------------------
# Gemma2 + Action Expert (two-expert) transformer
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Standard Gemma RMSNorm. weight is initialised to zero so (1 + w) = 1."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x32 = x32 * rms * (1.0 + self.weight.float())
        return x32.to(dtype)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm matching the checkpoint's `dense: Linear(cond, 3*dim)`.

    Returns (normed * (1 + scale) + shift, gate). The caller applies `gate` to
    the residual branch (attn output or MLP output).
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.dense = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x32 = x32 * rms                                  # [B, T, dim] fp32
        modulation = self.dense(cond).float()            # [B, 3*dim]
        if x32.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        x32 = x32 * (1.0 + scale) + shift
        return x32.to(dtype), gate.to(dtype)


class GemmaMLP(nn.Module):
    """Gated MLP: gelu(gate_proj(x)) * up_proj(x) -> down_proj."""

    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q: Tensor, k: Tensor, position_ids: Tensor,
               cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: [B, H, T, D]. position_ids: [B, T]. cos/sin: [S_max, D]."""
    cos_t = cos[position_ids].unsqueeze(1)              # [B, 1, T, D]
    sin_t = sin[position_ids].unsqueeze(1)

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    return q * cos_t + rotate_half(q) * sin_t, k * cos_t + rotate_half(k) * sin_t


class DualExpertBlock(nn.Module):
    """A single transformer layer that runs two experts in parallel.

    The two token streams (prefix from PaliGemma, suffix from action expert)
    have separate Q/K/V/O/MLP weights but their attention is computed jointly
    by concatenating along the sequence axis.

    The action-expert side optionally uses adaRMSNorm conditioned on the time
    embedding (pi05).
    """

    def __init__(self, paligemma: GemmaConfig, action_expert: GemmaConfig,
                 use_adarms_action: bool):
        super().__init__()
        self.paligemma_cfg = paligemma
        self.action_cfg = action_expert
        # PaliGemma side
        self.pg_pre_attn_norm = RMSNorm(paligemma.width)
        self.pg_post_attn_norm = RMSNorm(paligemma.width)
        self.pg_q = nn.Linear(paligemma.width, paligemma.num_heads * paligemma.head_dim, bias=False)
        self.pg_k = nn.Linear(paligemma.width, paligemma.num_kv_heads * paligemma.head_dim, bias=False)
        self.pg_v = nn.Linear(paligemma.width, paligemma.num_kv_heads * paligemma.head_dim, bias=False)
        self.pg_o = nn.Linear(paligemma.num_heads * paligemma.head_dim, paligemma.width, bias=False)
        self.pg_mlp = GemmaMLP(paligemma.width, paligemma.mlp_dim)

        # Action expert side. Uses the same head_dim/num_heads/num_kv_heads so
        # that Q/K/V can be concatenated along the sequence dimension before
        # the attention.
        self.use_adarms_action = use_adarms_action
        if use_adarms_action:
            self.ax_pre_attn_norm = AdaRMSNorm(action_expert.width, cond_dim=action_expert.width)
            self.ax_post_attn_norm = AdaRMSNorm(action_expert.width, cond_dim=action_expert.width)
        else:
            self.ax_pre_attn_norm = RMSNorm(action_expert.width)
            self.ax_post_attn_norm = RMSNorm(action_expert.width)
        self.ax_q = nn.Linear(action_expert.width, action_expert.num_heads * action_expert.head_dim, bias=False)
        self.ax_k = nn.Linear(action_expert.width, action_expert.num_kv_heads * action_expert.head_dim, bias=False)
        self.ax_v = nn.Linear(action_expert.width, action_expert.num_kv_heads * action_expert.head_dim, bias=False)
        self.ax_o = nn.Linear(action_expert.num_heads * action_expert.head_dim, action_expert.width, bias=False)
        self.ax_mlp = GemmaMLP(action_expert.width, action_expert.mlp_dim)

    def _norm_pre(self, x_pg: Tensor, x_ax: Tensor | None,
                  adarms_cond: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        h_pg = self.pg_pre_attn_norm(x_pg) if x_pg is not None else None
        gate_ax = None
        if x_ax is not None:
            if self.use_adarms_action:
                assert adarms_cond is not None, "adarms_cond required for pi05 action expert"
                h_ax, gate_ax = self.ax_pre_attn_norm(x_ax, adarms_cond)
            else:
                h_ax = self.ax_pre_attn_norm(x_ax)
        else:
            h_ax = None
        return h_pg, h_ax, gate_ax

    def _norm_post(self, x_pg: Tensor, x_ax: Tensor | None,
                   adarms_cond: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        h_pg = self.pg_post_attn_norm(x_pg) if x_pg is not None else None
        gate_ax = None
        if x_ax is not None:
            if self.use_adarms_action:
                h_ax, gate_ax = self.ax_post_attn_norm(x_ax, adarms_cond)
            else:
                h_ax = self.ax_post_attn_norm(x_ax)
        else:
            h_ax = None
        return h_pg, h_ax, gate_ax

    def forward(
        self,
        x_pg: Tensor | None,                  # [B, T_pg, width_pg]
        x_ax: Tensor | None,                  # [B, T_ax, width_ax]
        attn_mask_4d: Tensor,                 # [B, 1, T_q, T_kv]; 0 / -inf
        position_ids: Tensor,                 # [B, T_q]
        rope_cos: Tensor,
        rope_sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        adarms_cond: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, tuple[Tensor, Tensor] | None]:
        # ---- pre-attn norms ----
        h_pg, h_ax, gate_ax_pre = self._norm_pre(x_pg, x_ax, adarms_cond)

        # ---- project Q/K/V on each side, then concat along the sequence dim ----
        head_dim = self.paligemma_cfg.head_dim
        q_heads = self.paligemma_cfg.num_heads
        kv_heads = self.paligemma_cfg.num_kv_heads

        q_list, k_list, v_list = [], [], []
        if h_pg is not None:
            b, t_pg, _ = h_pg.shape
            q_list.append(self.pg_q(h_pg).view(b, t_pg, q_heads, head_dim))
            k_list.append(self.pg_k(h_pg).view(b, t_pg, kv_heads, head_dim))
            v_list.append(self.pg_v(h_pg).view(b, t_pg, kv_heads, head_dim))
        if h_ax is not None:
            b, t_ax, _ = h_ax.shape
            q_list.append(self.ax_q(h_ax).view(b, t_ax, q_heads, head_dim))
            k_list.append(self.ax_k(h_ax).view(b, t_ax, kv_heads, head_dim))
            v_list.append(self.ax_v(h_ax).view(b, t_ax, kv_heads, head_dim))

        q = torch.cat(q_list, dim=1).transpose(1, 2)    # [B, q_heads, T, D]
        k = torch.cat(k_list, dim=1).transpose(1, 2)
        v = torch.cat(v_list, dim=1).transpose(1, 2)

        q, k = apply_rope(q, k, position_ids, rope_cos, rope_sin)

        # KV cache: prepend stored prefix kv along the sequence axis.
        new_kv = (k, v) if use_cache else None
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # GQA: expand kv heads to match query heads.
        if kv_heads != q_heads:
            repeat = q_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_4d.to(q.dtype))
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, q_heads, D]
        attn_out = attn_out.view(attn_out.shape[0], attn_out.shape[1], q_heads * head_dim)

        # ---- split back per expert, project out, residual + post-norm + MLP ----
        cursor = 0
        out_pg = None
        out_ax = None
        if h_pg is not None:
            t_pg = h_pg.shape[1]
            attn_pg = self.pg_o(attn_out[:, cursor:cursor + t_pg])
            cursor += t_pg
            x_pg = x_pg + attn_pg
        if h_ax is not None:
            t_ax = h_ax.shape[1]
            attn_ax = self.ax_o(attn_out[:, cursor:cursor + t_ax])
            cursor += t_ax
            if self.use_adarms_action and gate_ax_pre is not None:
                x_ax = x_ax + gate_ax_pre * attn_ax
            else:
                x_ax = x_ax + attn_ax

        h_pg2, h_ax2, gate_ax_post = self._norm_post(
            x_pg if h_pg is not None else None,
            x_ax if h_ax is not None else None,
            adarms_cond,
        )
        if h_pg2 is not None:
            x_pg = x_pg + self.pg_mlp(h_pg2)
            out_pg = x_pg
        if h_ax2 is not None:
            mlp_ax = self.ax_mlp(h_ax2)
            if self.use_adarms_action and gate_ax_post is not None:
                x_ax = x_ax + gate_ax_post * mlp_ax
            else:
                x_ax = x_ax + mlp_ax
            out_ax = x_ax

        return out_pg, out_ax, new_kv


class DualExpertTransformer(nn.Module):
    """Stack of `DualExpertBlock`s plus shared embeddings/heads."""

    def __init__(self, cfg: Pi05Config, use_adarms_action: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_adarms_action = use_adarms_action
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.paligemma.width)
        self.layers = nn.ModuleList([
            DualExpertBlock(cfg.paligemma, cfg.action_expert, use_adarms_action)
            for _ in range(cfg.paligemma.depth)
        ])
        self.pg_final_norm = RMSNorm(cfg.paligemma.width)
        if use_adarms_action:
            self.ax_final_norm = AdaRMSNorm(cfg.action_expert.width, cond_dim=cfg.action_expert.width)
        else:
            self.ax_final_norm = RMSNorm(cfg.action_expert.width)

        # RoPE cache: prefix images (3*256=768) + prompt (200) + suffix actions (10)
        max_seq = 3 * 256 + cfg.max_token_len + cfg.action_horizon + 8
        cos, sin = build_rope_cache(max_seq, cfg.paligemma.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        prefix_embs: Tensor | None,
        suffix_embs: Tensor | None,
        attn_mask_4d: Tensor,
        position_ids: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        adarms_cond: Tensor | None = None,
    ) -> tuple[tuple[Tensor | None, Tensor | None], list[tuple[Tensor, Tensor]] | None]:
        x_pg = prefix_embs
        x_ax = suffix_embs
        new_kvs: list[tuple[Tensor, Tensor]] = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x_pg, x_ax, new_kv = layer(
                x_pg, x_ax, attn_mask_4d, position_ids,
                self.rope_cos, self.rope_sin,
                past_kv=past_kv, use_cache=use_cache, adarms_cond=adarms_cond,
            )
            if use_cache:
                new_kvs.append(new_kv)
        if x_pg is not None:
            x_pg = self.pg_final_norm(x_pg)
        if x_ax is not None:
            if self.use_adarms_action:
                x_ax, _ = self.ax_final_norm(x_ax, adarms_cond)
            else:
                x_ax = self.ax_final_norm(x_ax)
        return (x_pg, x_ax), new_kvs


# ---------------------------------------------------------------------------
# PI05 model
# ---------------------------------------------------------------------------


def sincos_time_embedding(t: Tensor, dim: int,
                          min_period: float = 4e-3, max_period: float = 4.0) -> Tensor:
    """t: [B] in [0, 1]. Returns [B, dim]."""
    assert dim % 2 == 0
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angles = (1.0 / period * 2.0 * math.pi).unsqueeze(0) * t.unsqueeze(1).float()
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(t.dtype)


def make_attn_2d_mask(pad_mask: Tensor, ar_mask: Tensor) -> Tensor:
    """pad_mask, ar_mask: [B, N] bool/int. Returns bool [B, N, N]."""
    cumsum = torch.cumsum(ar_mask.long(), dim=1)
    attn = cumsum.unsqueeze(1) <= cumsum.unsqueeze(2)
    valid = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return attn & valid


class HashTokenizer:
    """Tiny stand-in tokenizer for random-init runs.

    Produces deterministic token ids in [1, vocab_size) by hashing characters.
    Real PI05 uses the PaliGemma SentencePiece tokenizer; this is just a
    placeholder so the embedding lookup works end to end.
    """

    def __init__(self, vocab_size: int, max_len: int):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, prompt: str) -> tuple[Tensor, Tensor]:
        ids = [(ord(c) * 1315423911) % (self.vocab_size - 1) + 1 for c in prompt]
        ids = ids[: self.max_len]
        mask = [True] * len(ids) + [False] * (self.max_len - len(ids))
        ids = ids + [0] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
        )


class PI05Torch(nn.Module):
    """Pure-torch PI05 implementing the inference flow described in model.md."""

    IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __init__(self, cfg: Pi05Config | None = None):
        super().__init__()
        self.cfg = cfg or Pi05Config()
        # Vision projector must produce embeddings in the PaliGemma width.
        siglip_cfg = dataclasses.replace(self.cfg.siglip, projection_dim=self.cfg.paligemma.width)
        self.vision = SiglipVisionEncoder(siglip_cfg)
        self.transformer = DualExpertTransformer(self.cfg, use_adarms_action=True)

        self.action_in_proj = nn.Linear(self.cfg.action_dim, self.cfg.action_expert.width)
        self.action_out_proj = nn.Linear(self.cfg.action_expert.width, self.cfg.action_dim)
        self.time_mlp_in = nn.Linear(self.cfg.action_expert.width, self.cfg.action_expert.width)
        self.time_mlp_out = nn.Linear(self.cfg.action_expert.width, self.cfg.action_expert.width)

        self.tokenizer = HashTokenizer(self.cfg.vocab_size, self.cfg.max_token_len)

    # ------------------------------------------------------------------
    # Preprocessing: dict-style observation -> normalised tensors on device
    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _preprocess(self, obs: dict) -> dict:
        device = self._device()
        # Map entrypoint.py keys to the 3 SigLIP image keys.
        base = obs["observation/image"]
        wrist = obs["observation/wrist_image"]
        right_wrist = torch.zeros_like(base)
        raw_images = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        images_bchw: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for k, img in raw_images.items():
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0 * 2.0 - 1.0
            else:
                img = img.float()
            if img.shape[-1] == 3:                       # [B, H, W, C] -> [B, C, H, W]
                img = img.permute(0, 3, 1, 2).contiguous()
            images_bchw[k] = img.to(device)
            mask_val = 0.0 if k == "right_wrist_0_rgb" else 1.0
            masks[k] = torch.full((img.shape[0],), mask_val, dtype=torch.bool, device=device)

        state = obs["observation/state"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        # Pad state to action_dim (PI05 right-pads state into the action_dim slots).
        if state.shape[-1] < self.cfg.action_dim:
            pad = torch.zeros(state.shape[0], self.cfg.action_dim - state.shape[-1],
                              dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        state = state.float().to(device)

        prompt_ids, prompt_mask = self.tokenizer(obs.get("prompt", ""))
        return {
            "images": images_bchw,
            "image_masks": masks,
            "state": state,
            "tokenized_prompt": prompt_ids.to(device),
            "tokenized_prompt_mask": prompt_mask.to(device),
        }

    # ------------------------------------------------------------------
    # Mirrors pi0.embed_prefix
    # ------------------------------------------------------------------
    def embed_prefix(self, images: dict[str, Tensor], image_masks: dict[str, Tensor],
                     prompt_ids: Tensor, prompt_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        ar_mask: list[int] = []
        for key in self.IMAGE_KEYS:
            img_emb = self.vision(images[key])           # [B, 256, 2048]
            embs.append(img_emb)
            b, n = img_emb.shape[:2]
            pad_masks.append(image_masks[key].unsqueeze(1).expand(b, n))
            ar_mask += [0] * n

        lang_emb = self.transformer.embed_tokens(prompt_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(prompt_mask)
        ar_mask += [0] * lang_emb.shape[1]

        embs_cat = torch.cat(embs, dim=1)
        pad_cat = torch.cat(pad_masks, dim=1)
        ar = torch.tensor(ar_mask, dtype=torch.long, device=embs_cat.device)
        ar = ar.unsqueeze(0).expand(pad_cat.shape[0], -1)
        return embs_cat, pad_cat, ar

    # ------------------------------------------------------------------
    # Mirrors pi0.embed_suffix (pi05 branch: adaRMS, no state token)
    # ------------------------------------------------------------------
    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weight_dtype = self.action_in_proj.weight.dtype
        action_emb = self.action_in_proj(noisy_actions.to(weight_dtype))   # [B, H, 1024]

        time_emb = sincos_time_embedding(timestep, self.cfg.action_expert.width).to(weight_dtype)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = F.silu(time_emb)                                # [B, 1024]

        b, h = action_emb.shape[:2]
        pad_mask = torch.ones(b, h, dtype=torch.bool, device=action_emb.device)
        ar_mask = torch.tensor([1] + [0] * (h - 1), dtype=torch.long, device=action_emb.device)
        ar_mask = ar_mask.unsqueeze(0).expand(b, -1)
        return action_emb, pad_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------
    # Inference: 10-step Euler flow matching
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, obs: dict, *, num_steps: int = 10,
                       noise: Tensor | None = None) -> Tensor:
        proc = self._preprocess(obs)
        device = proc["state"].device

        # ---- prefix forward, build KV cache ----
        prefix_embs, prefix_pad, prefix_ar = self.embed_prefix(
            proc["images"], proc["image_masks"],
            proc["tokenized_prompt"], proc["tokenized_prompt_mask"],
        )
        prefix_attn_2d = make_attn_2d_mask(prefix_pad, prefix_ar)
        prefix_position_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1
        prefix_attn_4d = self._mask_to_additive(prefix_attn_2d)

        (_, _), kv_cache = self.transformer.forward(
            prefix_embs=prefix_embs, suffix_embs=None,
            attn_mask_4d=prefix_attn_4d, position_ids=prefix_position_ids,
            past_key_values=None, use_cache=True, adarms_cond=None,
        )

        # ---- denoise loop ----
        bsize = proc["state"].shape[0]
        if noise is None:
            noise = torch.randn(bsize, self.cfg.action_horizon, self.cfg.action_dim, device=device)
        x_t = noise.float()
        dt = -1.0 / num_steps
        time = torch.tensor(1.0, device=device)
        # `time >= -dt/2` is equivalent to "time > 0 with floating-point slack".
        while time.item() >= -dt / 2:
            t_b = time.expand(bsize)
            v_t = self._denoise_step(proc["state"], x_t, t_b, prefix_pad, kv_cache)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def _denoise_step(self, state: Tensor, x_t: Tensor, t_b: Tensor,
                      prefix_pad: Tensor, kv_cache: list[tuple[Tensor, Tensor]]) -> Tensor:
        # NB: PI05 doesn't feed `state` as a separate suffix token (it is folded
        # into the discrete prompt during training). We keep the signature here
        # for parity with pi0.py but don't use `state` inside the suffix.
        del state
        suffix_embs, suffix_pad, suffix_ar, adarms_cond = self.embed_suffix(x_t, t_b)

        b, t_suf = suffix_pad.shape
        t_pre = prefix_pad.shape[1]

        # suffix can attend to all of the (already-processed) prefix
        prefix_kv_mask = prefix_pad.unsqueeze(1).expand(b, t_suf, t_pre)
        suffix_self = make_attn_2d_mask(suffix_pad, suffix_ar)
        full_mask_2d = torch.cat([prefix_kv_mask, suffix_self], dim=2)
        full_mask_4d = self._mask_to_additive(full_mask_2d)

        prefix_offsets = prefix_pad.long().sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

        (_, x_ax), _ = self.transformer.forward(
            prefix_embs=None, suffix_embs=suffix_embs,
            attn_mask_4d=full_mask_4d, position_ids=position_ids,
            past_key_values=kv_cache, use_cache=False, adarms_cond=adarms_cond,
        )
        x_ax = x_ax[:, -self.cfg.action_horizon:]
        v = self.action_out_proj(x_ax.to(self.action_out_proj.weight.dtype))
        return v.float()

    @staticmethod
    def _mask_to_additive(mask_2d: Tensor) -> Tensor:
        return torch.where(mask_2d.unsqueeze(1), 0.0, torch.finfo(torch.float32).min)

    # ------------------------------------------------------------------
    # Weight loading from the HuggingFace-style safetensors checkpoint shipped
    # with openpi (`pi05_libero_pytorch/model.safetensors`).
    # ------------------------------------------------------------------
    def load_pretrained(self, safetensors_path: str, strict: bool = True) -> dict:
        """Load weights from the HF-format PI05 safetensors into this module.

        Returns the dict of parameter names that were successfully copied.
        """
        from safetensors import safe_open

        depth_pg = self.cfg.paligemma.depth
        rename: dict[str, str] = {}

        # ----- vision tower -----
        vt = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        rename[f"{vt}.embeddings.patch_embedding.weight"] = "vision.patch_embedding.weight"
        rename[f"{vt}.embeddings.patch_embedding.bias"]   = "vision.patch_embedding.bias"
        rename[f"{vt}.embeddings.position_embedding.weight"] = "vision.position_embedding.weight"
        rename[f"{vt}.post_layernorm.weight"] = "vision.post_layernorm.weight"
        rename[f"{vt}.post_layernorm.bias"]   = "vision.post_layernorm.bias"
        for i in range(self.cfg.siglip.depth):
            for ln in ("layer_norm1", "layer_norm2"):
                rename[f"{vt}.encoder.layers.{i}.{ln}.weight"] = f"vision.layers.{i}.{ln}.weight"
                rename[f"{vt}.encoder.layers.{i}.{ln}.bias"]   = f"vision.layers.{i}.{ln}.bias"
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                rename[f"{vt}.encoder.layers.{i}.self_attn.{proj}.weight"] = f"vision.layers.{i}.self_attn.{proj}.weight"
                rename[f"{vt}.encoder.layers.{i}.self_attn.{proj}.bias"]   = f"vision.layers.{i}.self_attn.{proj}.bias"
            for fc in ("fc1", "fc2"):
                rename[f"{vt}.encoder.layers.{i}.mlp.{fc}.weight"] = f"vision.layers.{i}.mlp.{fc}.weight"
                rename[f"{vt}.encoder.layers.{i}.mlp.{fc}.bias"]   = f"vision.layers.{i}.mlp.{fc}.bias"

        # ----- multi-modal projector (1152 -> 2048) -----
        mmp = "paligemma_with_expert.paligemma.model.multi_modal_projector"
        rename[f"{mmp}.linear.weight"] = "vision.multi_modal_projector.weight"
        rename[f"{mmp}.linear.bias"]   = "vision.multi_modal_projector.bias"

        # ----- PaliGemma language model (prefix expert) -----
        pg = "paligemma_with_expert.paligemma.model.language_model"
        for i in range(depth_pg):
            rename[f"{pg}.layers.{i}.input_layernorm.weight"]            = f"transformer.layers.{i}.pg_pre_attn_norm.weight"
            rename[f"{pg}.layers.{i}.post_attention_layernorm.weight"]   = f"transformer.layers.{i}.pg_post_attn_norm.weight"
            for src, dst in [("q_proj", "pg_q"), ("k_proj", "pg_k"),
                             ("v_proj", "pg_v"), ("o_proj", "pg_o")]:
                rename[f"{pg}.layers.{i}.self_attn.{src}.weight"] = f"transformer.layers.{i}.{dst}.weight"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rename[f"{pg}.layers.{i}.mlp.{proj}.weight"] = f"transformer.layers.{i}.pg_mlp.{proj}.weight"
        rename[f"{pg}.norm.weight"] = "transformer.pg_final_norm.weight"

        # ----- Token embedding (tied to paligemma.lm_head) -----
        rename["paligemma_with_expert.paligemma.lm_head.weight"] = "transformer.embed_tokens.weight"

        # ----- Action expert (suffix) -----
        ax = "paligemma_with_expert.gemma_expert.model"
        for i in range(depth_pg):
            for ln_src, ln_dst in [("input_layernorm", "ax_pre_attn_norm"),
                                   ("post_attention_layernorm", "ax_post_attn_norm")]:
                rename[f"{ax}.layers.{i}.{ln_src}.dense.weight"] = f"transformer.layers.{i}.{ln_dst}.dense.weight"
                rename[f"{ax}.layers.{i}.{ln_src}.dense.bias"]   = f"transformer.layers.{i}.{ln_dst}.dense.bias"
            for src, dst in [("q_proj", "ax_q"), ("k_proj", "ax_k"),
                             ("v_proj", "ax_v"), ("o_proj", "ax_o")]:
                rename[f"{ax}.layers.{i}.self_attn.{src}.weight"] = f"transformer.layers.{i}.{dst}.weight"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rename[f"{ax}.layers.{i}.mlp.{proj}.weight"] = f"transformer.layers.{i}.ax_mlp.{proj}.weight"
        rename[f"{ax}.norm.dense.weight"] = "transformer.ax_final_norm.dense.weight"
        rename[f"{ax}.norm.dense.bias"]   = "transformer.ax_final_norm.dense.bias"

        # ----- top-level heads -----
        for top in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            rename[f"{top}.weight"] = f"{top}.weight"
            rename[f"{top}.bias"]   = f"{top}.bias"

        own_state = self.state_dict()
        loaded: dict[str, str] = {}
        unmatched_src: list[str] = []
        with safe_open(safetensors_path, framework="pt") as f:
            ckpt_keys = set(f.keys())
            for src_name, dst_name in rename.items():
                if src_name not in ckpt_keys:
                    unmatched_src.append(src_name)
                    continue
                if dst_name not in own_state:
                    raise KeyError(f"Target parameter `{dst_name}` not found in PI05Torch")
                tensor = f.get_tensor(src_name)
                target = own_state[dst_name]
                if tensor.shape != target.shape:
                    raise ValueError(f"shape mismatch for {src_name} -> {dst_name}: "
                                     f"ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}")
                target.data.copy_(tensor.to(target.device).to(target.dtype))
                loaded[dst_name] = src_name

            ignored_ckpt = sorted(ckpt_keys - set(rename.keys()))

        # report
        missing_in_model = sorted(p for p in own_state if p not in loaded)
        if strict and (unmatched_src or missing_in_model):
            print("[load_pretrained] unmatched src:", unmatched_src[:5], f"({len(unmatched_src)} total)")
            print("[load_pretrained] missing in model:", missing_in_model[:5], f"({len(missing_in_model)} total)")
        else:
            print(f"[load_pretrained] loaded {len(loaded)} tensors; "
                  f"{len(missing_in_model)} model params untouched, "
                  f"{len(ignored_ckpt)} ckpt keys ignored")
        return {"loaded": loaded, "missing_in_model": missing_in_model, "ignored_ckpt": ignored_ckpt}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _build_dummy_obs() -> dict:
    return {
        "observation/image": torch.zeros((224, 224, 3), dtype=torch.uint8),
        "observation/wrist_image": torch.zeros((224, 224, 3), dtype=torch.uint8),
        "observation/state": torch.zeros((8,), dtype=torch.float32),
        "prompt": "pick up the red block on the table",
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pi05_torch] device={device}")

    model = PI05Torch().to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pi05_torch] params={n_params/1e6:.1f}M (random init)")

    obs = _build_dummy_obs()
    with torch.no_grad():
        action_chunk = model.sample_actions(obs)
    action_chunk = action_chunk[0].cpu()
    print(f"[pi05_torch] action_chunk.shape = {tuple(action_chunk.shape)}")
    print(action_chunk)


if __name__ == "__main__":
    main()
