"""Side-by-side comparison: JAX PI05 (entrypoint flow) vs pi05_torch.py.

This script:
  1. Loads the PI05 JAX policy with the libero checkpoint (`pi05_libero`).
  2. Runs the openpi input-transform pipeline on a dummy observation so we get
     the exact same `Observation` (images / tokenized prompt / state) that the
     JAX model would see.
  3. Calls `model.sample_actions(...)` directly with a deterministic noise of
     zeros — bypassing the random key — so the result is reproducible.
  4. Loads the same observation tensors into the pure-torch PI05 model from
     `pi05_torch.py`, copies weights from `model.safetensors`, runs the same
     denoise loop with the same zero noise.
  5. Reports max / mean abs diff between the 32-dim raw model outputs.

Run with the JAX venv (it has jax+torch+openpi): see __main__ for the env vars.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

# ---- openpi (JAX) imports
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.models import model as _model

# ---- pure-torch model
import pi05_torch


CONFIG_NAME = "pi05_libero"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


def build_dummy_obs():
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the red block on the table",
    }


def run_jax(obs_dict):
    """Returns (raw_action_32d_np, observation_dict_after_transforms)."""
    import jax
    import jax.numpy as jnp

    config = _config.get_config(CONFIG_NAME)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{CONFIG_NAME}")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # Replicate Policy.infer() up to the model call so we can capture the
    # exact transformed dict.
    inputs = jax.tree.map(lambda x: x, obs_dict)
    inputs = policy._input_transform(inputs)                     # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    obs = _model.Observation.from_dict(inputs)
    # Use a deterministic zero noise so the run is reproducible.
    noise = jnp.zeros((1, policy._model.action_horizon, policy._model.action_dim))  # noqa: SLF001
    rng = jax.random.key(0)
    raw = policy._model.sample_actions(rng, obs, noise=noise)    # noqa: SLF001
    raw_np = np.asarray(raw[0])

    # Also pull the transformed dict back out as numpy so torch can use it.
    transformed = jax.tree.map(lambda x: np.asarray(x), inputs)
    return raw_np, transformed, np.asarray(noise[0])


def run_torch(transformed: dict, noise: np.ndarray):
    """Runs pure-torch PI05 on the *already-transformed* JAX inputs.

    `transformed` matches what `Policy._input_transform` produced (image dict in
    [-1, 1] float32 NHWC, tokenized_prompt int32, etc.). We bypass the torch
    `_preprocess` and feed embed_prefix directly so the comparison isolates the
    network arithmetic.
    """
    cfg = pi05_torch.Pi05Config()
    model = pi05_torch.PI05Torch(cfg)
    model.load_pretrained(
        "/workspace/PI05_Weights/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors",
    )
    model = model.to(device=DEVICE, dtype=DTYPE)
    # Norm parameters that we keep in float32 in HF — keep them bfloat16 here
    # too since that's what the safetensors stored. The internal RMSNorm casts
    # back up to fp32 for the variance computation.
    model.eval()

    images = transformed["image"]
    image_masks = transformed["image_mask"]
    state = transformed["state"]
    prompt_ids = transformed["tokenized_prompt"]
    prompt_mask = transformed["tokenized_prompt_mask"]

    # Convert NHWC float32 in [-1, 1] -> NCHW
    def to_torch_image(arr):
        t = torch.from_numpy(np.asarray(arr)).to(DEVICE)
        t = t.permute(0, 3, 1, 2).contiguous()                   # NHWC -> NCHW
        return t.to(DTYPE)

    images_t = {k: to_torch_image(v) for k, v in images.items()}
    image_masks_t = {k: torch.from_numpy(np.asarray(v)).to(DEVICE) for k, v in image_masks.items()}
    state_t = torch.from_numpy(np.asarray(state)).to(DEVICE).to(torch.float32)
    prompt_ids_t = torch.from_numpy(np.asarray(prompt_ids)).to(DEVICE).long()
    prompt_mask_t = torch.from_numpy(np.asarray(prompt_mask)).to(DEVICE).bool()
    noise_t = torch.from_numpy(np.asarray(noise))[None].to(DEVICE).to(DTYPE)

    with torch.no_grad():
        # Replicate sample_actions but with already-prepared inputs
        prefix_embs, prefix_pad, prefix_ar = model.embed_prefix(
            images_t, image_masks_t, prompt_ids_t, prompt_mask_t,
        )
        prefix_attn_2d = pi05_torch.make_attn_2d_mask(prefix_pad, prefix_ar)
        prefix_position_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1
        prefix_attn_4d = model._mask_to_additive(prefix_attn_2d)  # noqa: SLF001

        prefix_embs = prefix_embs.to(DTYPE)
        (_, _), kv_cache = model.transformer.forward(
            prefix_embs=prefix_embs, suffix_embs=None,
            attn_mask_4d=prefix_attn_4d, position_ids=prefix_position_ids,
            past_key_values=None, use_cache=True, adarms_cond=None,
        )

        x_t = noise_t
        num_steps = 10
        dt = -1.0 / num_steps
        time = torch.tensor(1.0, device=DEVICE)
        while time.item() >= -dt / 2:
            t_b = time.expand(1)
            v_t = model._denoise_step(state_t, x_t, t_b, prefix_pad, kv_cache)  # noqa: SLF001
            x_t = x_t + dt * v_t
            time = time + dt
    return x_t[0].float().cpu().numpy()


def main():
    obs = build_dummy_obs()
    print(f"=== JAX run ({CONFIG_NAME}) ===")
    jax_out, transformed, noise = run_jax(obs)
    print("jax shape:", jax_out.shape)
    print("jax[0,:8]:", jax_out[0, :8])

    print(f"\n=== Torch run (pi05_torch.py, weights loaded) ===")
    torch_out = run_torch(transformed, noise)
    print("torch shape:", torch_out.shape)
    print("torch[0,:8]:", torch_out[0, :8])

    diff = np.abs(jax_out - torch_out)
    print(f"\n=== Diff ===")
    print(f"max abs diff:  {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")
    print(f"jax  norm: {np.linalg.norm(jax_out):.4f}")
    print(f"torch norm: {np.linalg.norm(torch_out):.4f}")
    print(f"per-step max diff: {diff.max(axis=1).round(4)}")

    print("\n=== Side-by-side first 7 dims (LiberoOutputs slice) ===")
    print("step  | JAX                              | Torch")
    for i in range(jax_out.shape[0]):
        j = "  ".join(f"{x:+.3f}" for x in jax_out[i, :7])
        t = "  ".join(f"{x:+.3f}" for x in torch_out[i, :7])
        print(f"{i:5d} | {j} | {t}")


if __name__ == "__main__":
    # Need to run with:
    #   OPENPI_DATA_HOME=/workspace/PI05_Weights \
    #   XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    #   XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    #   ./venv/bin/python compare_jax_vs_torch.py
    main()
