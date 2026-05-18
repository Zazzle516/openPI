import pickle
import torch

from pi05_infer import Pi05Inference

WEIGHTS_PKL = "/workspace/PI05_Weights/openpi-assets/checkpoints/pi05_libero_pytorch/weights.pkl"
NUM_VIEWS = 2
CHUNK_SIZE = 10

print(f"Loading converted weights from {WEIGHTS_PKL} ...")
with open(WEIGHTS_PKL, "rb") as f:
    checkpoint = pickle.load(f)

print("Building Pi05Inference (compiles Triton kernels + captures CUDA graph) ...")
policy = Pi05Inference(
    checkpoint=checkpoint,
    num_views=NUM_VIEWS,
    chunk_size=CHUNK_SIZE,
    discrete_state_input=False,
)

print("Running inference ...")
images = torch.zeros((NUM_VIEWS, 224, 224, 3), dtype=torch.bfloat16, device="cuda")
noise = torch.randn((CHUNK_SIZE, 32), dtype=torch.bfloat16, device="cuda")

with torch.no_grad():
    action_chunk = policy.forward(images, noise)

print("Success:")
print(action_chunk.shape, action_chunk.dtype)
print(action_chunk)

# python entrypoint.py
