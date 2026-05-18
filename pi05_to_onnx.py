"""Pure-PyTorch reference of Pi05Inference for ONNX export.

Mirrors the math of pi05_infer.py op-by-op. Triton kernels are replaced by
equivalent nn ops so the graph is exportable. Weights are random (architecture-
only export), and the 10-step Euler diffusion loop is unrolled inside the graph.

Inputs:
  images        : [num_views, 224, 224, 3] bfloat16 -> internally cast to fp32
  prompt_embeds : [prompt_len, 2048] bfloat16
  noise         : [chunk_size, 32] bfloat16
Output:
  action        : [chunk_size, 32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_VIEWS = 3
CHUNK_SIZE = 10
PROMPT_LEN = 200
NUM_STEPS = 10

VISION_DIM = 1152
VISION_FFN = 4304
VISION_LAYERS = 27
VISION_HEADS = 16
VISION_HEAD_DIM = VISION_DIM // VISION_HEADS  # 72
PATCH = 14
PATCHES_PER_SIDE = 224 // PATCH  # 16
PATCHES = PATCHES_PER_SIDE ** 2  # 256

ENC_DIM = 2048
ENC_FFN = 16384
ENC_LAYERS = 18
ENC_HEAD_DIM = 256
ENC_NUM_HEADS = 8
ENC_KV_HEADS = 1  # GQA: single KV head shared by 8 query heads

DEC_DIM = 1024
DEC_FFN = 4096
DEC_LAYERS = 18
ACTION_DIM = 32


def rope_table(max_pos: int, head_dim: int = ENC_HEAD_DIM) -> torch.Tensor:
    """Returns interleaved [cos, sin] table of shape [max_pos, head_dim]."""
    pos = torch.arange(max_pos, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    phase = pos[:, None] * inv_freq[None, :]
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    return torch.stack([cos, sin], dim=-1).reshape(max_pos, head_dim)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: [seq, n_heads, head_dim]. rope: [seq, head_dim] interleaved (cos,sin,cos,sin,...)."""
    seq, n_heads, head_dim = x.shape
    cos = rope[:, 0::2]  # [seq, head_dim/2]
    sin = rope[:, 1::2]
    x_pair = x.reshape(seq, n_heads, head_dim // 2, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    y0 = x0 * cos_b - x1 * sin_b
    y1 = x1 * cos_b + x0 * sin_b
    return torch.stack([y0, y1], dim=-1).reshape(seq, n_heads, head_dim)


class SigLIPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(VISION_DIM, eps=1e-5)
        self.qkv = nn.Linear(VISION_DIM, 3 * VISION_DIM, bias=True)
        self.proj = nn.Linear(VISION_DIM, VISION_DIM, bias=True)
        self.ln2 = nn.LayerNorm(VISION_DIM, eps=1e-5)
        self.fc1 = nn.Linear(VISION_DIM, VISION_FFN, bias=True)
        self.fc2 = nn.Linear(VISION_FFN, VISION_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256, 1152]
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(x.shape[0], x.shape[1], 3, VISION_HEADS, VISION_HEAD_DIM)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], VISION_DIM)
        x = x + self.proj(attn)
        h = self.ln2(x)
        h = F.gelu(self.fc1(h), approximate="tanh")
        x = x + self.fc2(h)
        return x


class SigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # NHWC patch embedding: [14,14,3,1152] reshapes to [588,1152]
        self.patch_w = nn.Parameter(torch.empty(PATCH * PATCH * 3, VISION_DIM))
        self.patch_b = nn.Parameter(torch.empty(VISION_DIM))
        self.pos_emb = nn.Parameter(torch.empty(PATCHES, VISION_DIM))
        self.blocks = nn.ModuleList([SigLIPBlock() for _ in range(VISION_LAYERS)])
        self.final_norm = nn.LayerNorm(VISION_DIM, eps=1e-5)
        self.proj_w = nn.Parameter(torch.empty(VISION_DIM, ENC_DIM))
        self.proj_b = nn.Parameter(torch.empty(ENC_DIM))
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [V, 224, 224, 3] in NHWC
        v = images.shape[0]
        # Unfold: [V, 224, 224, 3] -> [V, 16, 14, 16, 14, 3] -> permute -> [V, 16, 16, 14, 14, 3] -> [V, 256, 588]
        x = images.view(v, PATCHES_PER_SIDE, PATCH, PATCHES_PER_SIDE, PATCH, 3)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(v, PATCHES, PATCH * PATCH * 3)
        x = x @ self.patch_w + self.patch_b + self.pos_emb  # [V, 256, 1152]
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        x = x @ self.proj_w + self.proj_b  # [V, 256, 2048]
        return x


class GemmaEncoderBlock(nn.Module):
    """RMSNorm -> GQA (8 Q heads, 1 KV head, RoPE) -> RMSNorm -> GeGLU FFN."""

    def __init__(self):
        super().__init__()
        self.rms1_w = nn.Parameter(torch.ones(ENC_DIM))
        # qkv weight: [ENC_DIM, (num_heads + 2) * head_dim] = [2048, 2560]
        self.qkv_w = nn.Parameter(torch.empty(ENC_DIM, (ENC_NUM_HEADS + 2) * ENC_HEAD_DIM))
        self.o_w = nn.Parameter(torch.empty(ENC_NUM_HEADS * ENC_HEAD_DIM, ENC_DIM))
        self.rms2_w = nn.Parameter(torch.ones(ENC_DIM))
        self.gate_w = nn.Parameter(torch.empty(ENC_DIM, ENC_FFN))
        self.up_w = nn.Parameter(torch.empty(ENC_DIM, ENC_FFN))
        self.down_w = nn.Parameter(torch.empty(ENC_FFN, ENC_DIM))
        nn.init.xavier_uniform_(self.qkv_w)
        nn.init.xavier_uniform_(self.o_w)
        nn.init.xavier_uniform_(self.gate_w)
        nn.init.xavier_uniform_(self.up_w)
        nn.init.xavier_uniform_(self.down_w)

    @staticmethod
    def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Note: pi05_infer.py kernels apply factor without a learnable weight; we keep w=1 by default.
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + eps).to(x.dtype)) * w

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        seq = x.shape[0]
        h = self.rms_norm(x, self.rms1_w)
        qkv = h @ self.qkv_w  # [S, 2560]
        q = qkv[:, : ENC_NUM_HEADS * ENC_HEAD_DIM].reshape(seq, ENC_NUM_HEADS, ENC_HEAD_DIM)
        k = qkv[:, ENC_NUM_HEADS * ENC_HEAD_DIM : (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM].reshape(seq, 1, ENC_HEAD_DIM)
        v = qkv[:, (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM :].reshape(seq, 1, ENC_HEAD_DIM)
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # GQA attention, broadcast 1 KV head across 8 Q heads
        scale = ENC_HEAD_DIM ** -0.5
        logits = torch.einsum("shd,khd->shk", q, k) * scale  # [S, H, S]
        attn = F.softmax(logits, dim=-1)
        ctx = torch.einsum("shk,khd->shd", attn, v)  # [S, H, D]
        ctx = ctx.reshape(seq, ENC_NUM_HEADS * ENC_HEAD_DIM)
        x = x + ctx @ self.o_w
        h = self.rms_norm(x, self.rms2_w)
        gate = F.gelu(h @ self.gate_w, approximate="tanh")
        up = h @ self.up_w
        x = x + (gate * up) @ self.down_w
        return x


class GemmaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([GemmaEncoderBlock() for _ in range(ENC_LAYERS)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> tuple[torch.Tensor, list, list]:
        ks, vs = [], []
        for blk in self.blocks:
            seq = x.shape[0]
            h = blk.rms_norm(x, blk.rms1_w)
            qkv = h @ blk.qkv_w
            q = qkv[:, : ENC_NUM_HEADS * ENC_HEAD_DIM].reshape(seq, ENC_NUM_HEADS, ENC_HEAD_DIM)
            k = qkv[:, ENC_NUM_HEADS * ENC_HEAD_DIM : (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM].reshape(seq, 1, ENC_HEAD_DIM)
            v = qkv[:, (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM :].reshape(seq, 1, ENC_HEAD_DIM)
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)
            ks.append(k)
            vs.append(v)
            scale = ENC_HEAD_DIM ** -0.5
            logits = torch.einsum("shd,khd->shk", q, k) * scale
            attn = F.softmax(logits, dim=-1)
            ctx = torch.einsum("shk,khd->shd", attn, v).reshape(seq, ENC_NUM_HEADS * ENC_HEAD_DIM)
            x = x + ctx @ blk.o_w
            h = blk.rms_norm(x, blk.rms2_w)
            gate = F.gelu(h @ blk.gate_w, approximate="tanh")
            up = h @ blk.up_w
            x = x + (gate * up) @ blk.down_w
        return x, ks, vs


class GemmaDecoderBlock(nn.Module):
    """AdaRMSNorm + gating, GQA cross-attn over [prefix_K|suffix_K], GeGLU FFN."""

    def __init__(self):
        super().__init__()
        # AdaRMSNorm modulator: maps time_emb [1024] -> [3 * 1024] = (scale, shift, gate)
        self.pre_attn_mod_w = nn.Parameter(torch.empty(DEC_DIM, 3 * DEC_DIM))
        self.pre_attn_mod_b = nn.Parameter(torch.zeros(3 * DEC_DIM))
        self.qkv_w = nn.Parameter(torch.empty(DEC_DIM, (ENC_NUM_HEADS + 2) * ENC_HEAD_DIM))
        self.o_w = nn.Parameter(torch.empty(ENC_NUM_HEADS * ENC_HEAD_DIM, DEC_DIM))
        self.pre_ffn_mod_w = nn.Parameter(torch.empty(DEC_DIM, 3 * DEC_DIM))
        self.pre_ffn_mod_b = nn.Parameter(torch.zeros(3 * DEC_DIM))
        self.gate_w = nn.Parameter(torch.empty(DEC_DIM, DEC_FFN))
        self.up_w = nn.Parameter(torch.empty(DEC_DIM, DEC_FFN))
        self.down_w = nn.Parameter(torch.empty(DEC_FFN, DEC_DIM))
        for p in [self.pre_attn_mod_w, self.qkv_w, self.o_w, self.pre_ffn_mod_w,
                  self.gate_w, self.up_w, self.down_w]:
            nn.init.xavier_uniform_(p)

    @staticmethod
    def adarms(x: torch.Tensor, time_emb: torch.Tensor, mod_w: torch.Tensor, mod_b: torch.Tensor,
               eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [S, 1024]. time_emb: [S, 1024]. style: [S, 3072].
        style = time_emb @ mod_w + mod_b
        s_scale, s_shift, s_gate = style.chunk(3, dim=-1)
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + eps).to(x.dtype)
        return x_norm * (1.0 + s_scale) + s_shift, s_gate

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, rope_dec: torch.Tensor,
                prefix_k_list: list, prefix_v_list: list, layer_idx: int,
                prefix_len: int) -> torch.Tensor:
        seq = x.shape[0]
        x_normed, gate_attn = self.adarms(x, time_emb, self.pre_attn_mod_w, self.pre_attn_mod_b)
        qkv = x_normed @ self.qkv_w
        q = qkv[:, : ENC_NUM_HEADS * ENC_HEAD_DIM].reshape(seq, ENC_NUM_HEADS, ENC_HEAD_DIM)
        k_suf = qkv[:, ENC_NUM_HEADS * ENC_HEAD_DIM : (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM].reshape(seq, 1, ENC_HEAD_DIM)
        v_suf = qkv[:, (ENC_NUM_HEADS + 1) * ENC_HEAD_DIM :].reshape(seq, 1, ENC_HEAD_DIM)
        q = apply_rope(q, rope_dec)
        k_suf = apply_rope(k_suf, rope_dec)
        # Concat prefix (encoder cache, already RoPE'd) and suffix
        k_pref = prefix_k_list[layer_idx]  # [prefix_len, 1, head_dim]
        v_pref = prefix_v_list[layer_idx]
        k_full = torch.cat([k_pref, k_suf], dim=0)
        v_full = torch.cat([v_pref, v_suf], dim=0)
        scale = ENC_HEAD_DIM ** -0.5
        logits = torch.einsum("shd,khd->shk", q, k_full) * scale
        attn = F.softmax(logits, dim=-1)
        ctx = torch.einsum("shk,khd->shd", attn, v_full).reshape(seq, ENC_NUM_HEADS * ENC_HEAD_DIM)
        x = x + (ctx @ self.o_w) * gate_attn
        x_normed, gate_ffn = self.adarms(x, time_emb, self.pre_ffn_mod_w, self.pre_ffn_mod_b)
        gate = F.gelu(x_normed @ self.gate_w, approximate="tanh")
        up = x_normed @ self.up_w
        x = x + ((gate * up) @ self.down_w) * gate_ffn
        return x


class Pi05Module(nn.Module):
    """Full pi05 forward, 10-step Euler diffusion unrolled."""

    def __init__(self):
        super().__init__()
        self.vision = SigLIP()
        self.encoder = GemmaEncoder()

        # Time MLP (sincos -> linear -> swish -> linear -> swish), one per step precomputed
        self.time_embeds = nn.Parameter(torch.zeros(NUM_STEPS, DEC_DIM))  # base (would be sincos in real ckpt)
        self.time_in_w = nn.Parameter(torch.empty(DEC_DIM, DEC_DIM))
        self.time_in_b = nn.Parameter(torch.zeros(DEC_DIM))
        self.time_out_w = nn.Parameter(torch.empty(DEC_DIM, DEC_DIM))
        self.time_out_b = nn.Parameter(torch.zeros(DEC_DIM))

        self.action_in_w = nn.Parameter(torch.empty(ACTION_DIM, DEC_DIM))
        self.action_in_b = nn.Parameter(torch.zeros(DEC_DIM))
        self.dec_blocks = nn.ModuleList([GemmaDecoderBlock() for _ in range(DEC_LAYERS)])
        self.final_mod_w = nn.Parameter(torch.empty(DEC_DIM, 3 * DEC_DIM))
        self.final_mod_b = nn.Parameter(torch.zeros(3 * DEC_DIM))
        # Pre-scaled by -1/num_steps as in pi05_infer.py:670
        self.action_out_w = nn.Parameter(torch.empty(DEC_DIM, ACTION_DIM))
        self.action_out_b = nn.Parameter(torch.zeros(ACTION_DIM))

        for p in [self.time_in_w, self.time_out_w, self.action_in_w,
                  self.final_mod_w, self.action_out_w]:
            nn.init.xavier_uniform_(p)

        # Precomputed RoPE tables (registered as buffers so they go into the graph as constants)
        prefix_len = NUM_VIEWS * PATCHES + PROMPT_LEN
        self.register_buffer("rope_enc", rope_table(prefix_len), persistent=False)
        # Decoder positions start at prefix_len-1 (last prompt token) for chunk_size positions
        max_pos = prefix_len - 1 + CHUNK_SIZE
        full = rope_table(max_pos + 1)
        self.register_buffer("rope_dec", full[prefix_len - 1 : prefix_len - 1 + CHUNK_SIZE], persistent=False)

    def forward(self, images: torch.Tensor, prompt_embeds: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # images: [V, 224, 224, 3] ; prompt_embeds: [P, 2048] ; noise: [chunk, 32]
        # 1) Vision -> [V, 256, 2048] -> flatten to [V*256, 2048]
        vis = self.vision(images)
        vis = vis.reshape(NUM_VIEWS * PATCHES, ENC_DIM)
        # 2) Concat with prompt to form encoder input [prefix_len, 2048]
        prefix = torch.cat([vis, prompt_embeds], dim=0)
        # 3) Run prefix through Gemma encoder, capture per-layer K/V cache
        _, prefix_k, prefix_v = self.encoder(prefix, self.rope_enc)

        # 4) Diffusion: 10 unrolled Euler steps
        x_t = noise
        for step in range(NUM_STEPS):
            # Time embedding for this step: time_embeds[step] -> in_proj+silu -> out_proj+silu
            te = self.time_embeds[step].unsqueeze(0)  # [1, 1024]
            te = F.silu(te @ self.time_in_w + self.time_in_b)
            te = F.silu(te @ self.time_out_w + self.time_out_b)
            te = te.expand(CHUNK_SIZE, -1)  # [chunk, 1024]
            # Action input projection
            h = x_t @ self.action_in_w + self.action_in_b  # [chunk, 1024]
            for i, blk in enumerate(self.dec_blocks):
                h = blk(h, te, self.rope_dec, prefix_k, prefix_v, i, prefix.shape[0])
            # Final AdaRMSNorm + projector to action
            x_normed, _ = GemmaDecoderBlock.adarms(h, te, self.final_mod_w, self.final_mod_b)
            v_t = x_normed @ self.action_out_w + self.action_out_b  # [chunk, 32]
            # Pi05 absorbs (-1/num_steps) into action_out_w/b, so update is x_t = x_t + v_t
            # (i.e. v_t already equals -(1/N) * raw_v). We mirror that here for parity.
            x_t = x_t + v_t
        return x_t


def export(out_path: str = "pi05.onnx", opset: int = 17, device: str = "cuda"):
    torch.manual_seed(0)
    model = Pi05Module().to(device).eval()
    images = torch.randn(NUM_VIEWS, 224, 224, 3, device=device)
    prompt_embeds = torch.randn(PROMPT_LEN, ENC_DIM, device=device)
    noise = torch.randn(CHUNK_SIZE, ACTION_DIM, device=device)

    with torch.no_grad():
        out = model(images, prompt_embeds, noise)
    print(f"forward OK, output shape={tuple(out.shape)}")

    torch.onnx.export(
        model,
        (images, prompt_embeds, noise),
        out_path,
        input_names=["images", "prompt_embeds", "noise"],
        output_names=["action"],
        opset_version=opset,
        dynamic_shapes={
            "images": None,
            "prompt_embeds": {0: torch.export.Dim("prompt_len", min=1, max=PROMPT_LEN)},
            "noise": None,
        },
        dynamo=True,
        external_data=True,
    )
    print(f"wrote {out_path} (+ external data sidecar)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="pi05.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    export(args.out, args.opset, args.device)
