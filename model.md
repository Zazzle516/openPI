Assuming your active `pi05_libero` config:

```python
Pi0Config(
    pi05=True,
    action_horizon=10,
    action_dim=32,
    discrete_state_input=False,
)
```

Default variants are:

```text
PaliGemma expert: gemma_2b    width=2048, depth=18, mlp_dim=16384
Action expert:   gemma_300m  width=1024, depth=18, mlp_dim=4096
Vision encoder:  SigLIP So400m/14, width=1152, depth=27, mlp_dim=4304
```

Your current [pi0.py](/home/zazzle/openpi/src/openpi/models/pi0.py:66) is locally edited to be PI05-oriented: it always creates `time_mlp_in/out`, and `compute_loss()` is currently `pass`.

```mermaid
flowchart TD
    A["Observation after Policy preprocessing<br/>images: 3 x [B,224,224,3]<br/>state: [B,32]<br/>tokenized_prompt: [B,200]"] --> B["SigLIP image encoder<br/>shared for each image"]

    B --> B1["Patch conv 14x14 stride 14<br/>[B,224,224,3] -> [B,16,16,1152]"]
    B1 --> B2["Flatten + pos emb<br/>[B,256,1152]"]
    B2 --> B3["27 x SigLIP Transformer blocks<br/>LN -> MHA 16 heads -> residual<br/>LN -> MLP 1152->4304->1152 -> residual<br/>GeLU inside MLP"]
    B3 --> B4["Dense head 1152->2048<br/>per image: [B,256,2048]"]
    B4 --> C["3 images concat<br/>[B,768,2048]"]

    A --> D["PaliGemma token embedder<br/>vocab=257152, dim=2048<br/>prompt [B,200] -> [B,200,2048]"]

    C --> E["Prefix tokens"]
    D --> E
    E --> E1["prefix_tokens<br/>[B,968,2048]<br/>= 3*256 image tokens + 200 prompt tokens"]

    F["Initial noisy actions<br/>[B,10,32]"] --> G["action_in_proj<br/>32 -> 1024<br/>[B,10,1024]"]

    H["timestep t<br/>[B]"] --> H1["sincos posemb<br/>[B,1024]"]
    H1 --> H2["time_mlp_in<br/>1024 -> 1024"]
    H2 --> H3["swish"]
    H3 --> H4["time_mlp_out<br/>1024 -> 1024"]
    H4 --> H5["swish<br/>adarms_cond [B,1024]"]

    G --> I["Suffix/action tokens<br/>[B,10,1024]"]
    H5 --> J["18 x two-expert Gemma blocks"]
    E1 --> J
    I --> J

    J --> K["suffix_out last action tokens<br/>[B,10,1024]"]
    K --> L["action_out_proj<br/>1024 -> 32<br/>v_t [B,10,32]"]
    L --> M["Euler denoise step<br/>x_t = x_t + dt * v_t<br/>repeat num_steps=10"]
    M --> N["Predicted normalized action chunk<br/>[B,10,32]"]
```

Inside the **18 Gemma blocks** from [gemma.py](/home/zazzle/openpi/src/openpi/models/gemma.py:284), each layer has two streams:

```text
Expert 0: PaliGemma/image-language stream
tokens: [B, prefix_len, 2048]

Expert 1: action stream
tokens: [B, action_horizon, 1024]
```

Each block does:

```text
RMSNorm / adaRMSNorm
-> multi-head attention over concatenated prefix + action tokens
-> residual
-> RMSNorm / adaRMSNorm
-> gated MLP
-> residual
```

Attention details from [gemma.py:157](/home/zazzle/openpi/src/openpi/models/gemma.py:157):

```text
num_heads = 8
num_kv_heads = 1
head_dim = 256

Query shape per stream:
PaliGemma: [B,T,8,256]
Action:    [B,T,8,256]

K/V shape per stream:
PaliGemma: [B,T,1,256]
Action:    [B,T,1,256]
```

The streams have different hidden widths, but attention projects both into the same head space, concatenates along token length, attends jointly, then projects back to each stream’s own width.

The Gemma MLP from [lora.py:88](/home/zazzle/openpi/src/openpi/models/lora.py:88) is a gated GeLU MLP:

```text
x -> Linear(features -> hidden_dim) -> GeLU
x -> Linear(features -> hidden_dim)
multiply both branches
-> Linear(hidden_dim -> features)
```

So per block:

```text
PaliGemma MLP: 2048 -> 16384 -> 2048, GeLU gate
Action MLP:    1024 -> 4096  -> 1024, GeLU gate
```

For PI05 specifically, the action expert uses **adaRMSNorm**. The timestep MLP produces `adarms_cond: [B,1024]`, and each action-expert RMSNorm creates:

```text
scale, shift, gate = Dense(1024 -> 3*1024)(adarms_cond)
```

That gate modulates the residual:

```python
x + y * gate
```

So the short architecture summary is:

```text
3 RGB views -> SigLIP So400m/14 -> 768 visual tokens of dim 2048
prompt -> PaliGemma embedder -> 200 language tokens of dim 2048
actions/noise -> Linear -> 10 action tokens of dim 1024
timestep -> sincos -> 2-layer swish MLP -> adaRMS conditioning
18 two-expert Gemma layers -> action hidden states
Linear 1024->32 -> denoising velocity
10 denoising steps -> [10,32] normalized action chunk
```