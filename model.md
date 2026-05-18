# Model

## SigLIP

### Pipeline

输入: 3 个 [224, 224] 的图片，Channel = Main Camera + Right Hand + Left Hand = 3
> Tip: Right Hand cound be empty

**Unfold & Matmul**: (NHWC)

- input_shape = [B, 224, 224, 3] => (view + permute) => [B, 16, 16, 14, 14, 3]
- kernel_size = patch_size = 14 x 14
- stride = 14
- N = patches number of each pics = (224/14) x (224/14) = 256

- unfold patch [B, 256, 14x14x3=588]
- matmul: [B, 256, 588] x Weight [588, 1152] + bias [1152, ]
- output_shape = [B, 256, 1152]

**Position Embedding**

- Look-Up Table: [N, 1152], Element-wise Addition

**Transformer Block x 27**:

- Block Input = [B, N, 1152]

- Layer Norm for MHSA: [B, N, 1152] => [B, N, 1152]
  
- MHSA(Multi-Head Self Attention)
  - num_heads = 16
  - Attention Map = [B, Head_Dim, N, N]
  - output_shape = [B, N, 1152]

- Residual Addition: [B, N, 1152] => [B, N, 1152]

- Layer Norm for MLP: [B, N, 1152] => [B, N, 1152]

- MLP / FFN
  - Linear: [B, 256, 1152] => [B, 256, 4304]
  - GeLU
  - Linear: [B, 256, 4304] => [B, 256, 1152]

**Post Layer Norm**: [B, N, 1152] => [B, N, 1152]

**End**


### Diagram

```mermaid
graph TD
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef embed fill:#fff3e0,stroke:#ffb300,stroke-width:2px;
    classDef transformer fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef pool fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#e53935,stroke-width:2px;

    %% 数据流
    A[Input Image NHWC <br> Shape: B x 224 x 224 x 3]:::input --> B(Unfold into Patches <br> view + permute<br> patch_size=14, stride=14 <br> equals to B x 256 x 588):::embed

    B --> C(Fused Patch Embedding + Position Embedding <br> matmul_small_bias_res_mod kernel <br>Output Shape: B x 256 x 1152):::embed

    %% Transformer Blocks
    subgraph SiglipEncoder [ViT-So400M: Transformer Block x 27]
        direction TB
        E1[Block Input: B x 256 x 1152] --> F1(LayerNorm)
        F1 --> G1(Multi-Head Self Attention <br> Heads=16, HeadDim=72, Dim=1152)
        G1 --> H1{Residual Add}
        E1 --> H1

        H1 --> I1(LayerNorm)
        I1 --> J1(MLP / FFN <br> 1152 -> 4304 -> 1152, Act=GELU)
        J1 --> K1{Residual Add}
        H1 --> K1
    end

    C --> SiglipEncoder:::transformer

    %% 输出
    SiglipEncoder --> L(Final LayerNorm <br> vision_final_norm <br> Output Shape: B x 256 x 1152):::pool
    L --> M(Multi-Modal Projector <br> Linear: 1152 -> 2048 + bias <br> Output Shape: B x 256 x 2048):::pool
    M --> N[Vision Embeddings keep all 256 patch token <br> Output Shape: B x 256 x 2048]:::output
```

### Operators Involved

matmul

Element-Wise: add, sub, mul, div

exp, rsqrt, sigmoid

reduce-sum, reduce-max (归约求最大值)

load, store

cast(bf16↔fp32)

index-compute, mask


## Gemma2



## Overview

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