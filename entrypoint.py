import os
import torch
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

CONFIG_NAME = "pi05_libero"
print(f"Fetch arguments: {CONFIG_NAME} ...")
config = _config.get_config(CONFIG_NAME)
checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{CONFIG_NAME}")

print("Loading...")

# JAX model
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

print("Loading Complete, Inference...")

# 4. 构造推理输入 (Observation)
dummy_obs = {
    # base_0_rgb
    "observation/image": torch.zeros((224, 224, 3), dtype=torch.uint8),
    # left_wrist_0_rgb, right_wrist_0_rgb(0)
    "observation/wrist_image": torch.zeros((224, 224, 3), dtype=torch.uint8),
    # state / padding / normalization
    "observation/state": torch.zeros((8,), dtype=torch.float32),
    "prompt": "pick up the red block on the table",
}

# 5. 执行推理 (Flow Matching 动作生成)
with torch.no_grad():
    action_chunk = policy.infer(dummy_obs)

print("Success:")
print(action_chunk)

# OPENPI_DATA_HOME=/mnt/e/pi_checkpoints XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 XLA_PYTHON_CLIENT_ALLOCATOR=platform python entrypoint.py
