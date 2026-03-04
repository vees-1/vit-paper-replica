#  `An Image is Worth 16×16 Words`

Paper: Dosovitskiy et al. (2021) — [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)   
Conference: ICLR 2021

---

## `Architecture`

The ViT architecture treats an image as a sequence of fixed-size patches, processing them through a standard Transformer encoder. This replication implements each sub-component modularly, mapping directly to the paper's equations.

```
Input Image (224×224×3)
        │
        ▼
┌──────────────────────┐
│   Patch Embedding    │  Conv2d(kernel=16, stride=16) → [N, 768]
│   (Equation 1)       │  + CLS Token + Position Embedding
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Transformer Encoder │  × 12 layers (ViT-Base)
│  ┌────────────────┐  │
│  │  LN → MSA      │  │  Equation 2: Multi-Head Self-Attention block
│  │  + Residual    │  │
│  ├────────────────┤  │
│  │  LN → MLP      │  │  Equation 3: MLP block (GELU activation)
│  │  + Residual    │  │
│  └────────────────┘  │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  LN → CLS Token      │  Equation 4: Classification head
│  → Linear Classifier │
└──────────────────────┘
        │
        ▼
   Class Logits
```


```
   Model Variant
        │
        ▼

| Model     | Layers | Hidden Size     | MLP Size | Heads | Params |
|-----------|--------|-----------------|----------|-------|--------|
| ViT-Base  | 12     | 768             | 3072     | 12    | ~86M   |
```

---

## `Implementation Details`

### `PatchEmbedding`
Implements `Equation 1` of the paper. A 2D image of shape `(H, W, C)` is converted into a sequence of `N = HW/P²` flattened patch embeddings of dimension `D` using a `nn.Conv2d` layer with `kernel_size=P` and `stride=P`. A learnable `[class]` token is prepended, and learnable 1D positional embeddings are added.

```python
self.patcher = nn.Conv2d(in_channels=3, out_channels=768,
                          kernel_size=16, stride=16, padding=0)
```

### `MultiheadSelfAttentionBlock`
Implements `Equation 2`. Each block applies LayerNorm before the attention operation, followed by a residual connection:

```
z'_l = MSA(LN(z_{l-1})) + z_{l-1}
```

Uses `nn.MultiheadAttention` with `embed_dim=768`, `num_heads=12`.

### `MLPBlock`
Implements `Equation 3`. Two linear layers with GELU activation and dropout, wrapped in LayerNorm with a residual connection:

```
z_l = MLP(LN(z'_l)) + z'_l
```

Architecture: `Linear(768 → 3072) → GELU → Dropout → Linear(3072 → 768) → Dropout`

### `TransformerEncoderBlock`
Combines the MSA block and MLP block into a single encoder layer. Verified against PyTorch's native `nn.TransformerEncoderLayer` (with `norm_first=True`, `activation="gelu"`) to confirm architectural equivalence.

### `ViT`
Assembles all components into the complete ViT architecture. Implements `Equation 4` — the output representation `y` is taken from the `[class]` token at position 0 after the final LayerNorm, then passed through a classification head.

---

## `Hyperparameters`

Training from scratch follows the ViT paper's Table 3 specifications for ImageNet-scale training (adapted for the small dataset regime):
```
| Hyperparameter     | Value                          |
|--------------------|--------------------------------|
| Image size         | 224 × 224                      |
| Patch size         | 16 × 16                        |
| Optimizer          | Adam (β₁=0.9, β₂=0.999)        |
| Learning rate      | 3e-3 (scratch), 1e-3 (finetune)|
| Weight decay       | 0.3                            |
| Batch size         | 32                             |
| Activation         | GELU                           |
| Dropout            | 0.1 (MLP layers)               |
| Loss function      | CrossEntropyLoss               |
```
---

## `Experiments`

### `Experiment 1: Training from Scratch`
ViT-Base trained from scratch on the pizza/steak/sushi 3-class dataset using Adam with the paper's recommended hyperparameters. As expected from the paper's findings (Section 4.3), ViT underperforms on small datasets due to its lack of CNN-style inductive biases (translation equivariance, locality).

### `Experiment 2: Transfer Learning with Pretrained Weights`
`torchvision.models.vit_b_16` pretrained on ImageNet-21k was loaded, all base parameters were frozen, and only the classification head was replaced with a zero-initialized `nn.Linear(768 → num_classes)` layer and fine-tuned. This mirrors the paper's fine-tuning protocol (Section 3.2).

This experiment demonstrates the core thesis of the paper: large-scale pre-training enables ViT to achieve strong performance even on small downstream datasets with minimal compute.

---

## `Requirements`

Install dependencies:

```bash
pip install torch torchvision torchinfo matplotlib numpy Pillow
```

---

## `Usage`

Open and run `vit.ipynb` sequentially. The notebook is organized into the following sections:

1. Data Loading: Downloads and prepares the pizza/steak/sushi dataset
2. Patch Visualization: Visualizes how images are divided into 16×16 patches
3. Component Implementation: Builds `PatchEmbedding`, `MultiheadSelfAttentionBlock`, `MLPBlock`, `TransformerEncoderBlock`, and `ViT` step by step
4. Architecture Verification: Validates shapes and parameter counts against Table 1 of the paper using `torchinfo`
5. Training from Scratch: Full training loop with loss curve visualization
6. Transfer Learning: Fine-tunes pretrained `ViT-B/16` on the downstream task

---

## `Key Findings`

- The custom `ViT` implementation produces the expected `~86M` parameter count matching ViT-Base from Table 1 of the paper.
- The custom `TransformerEncoderBlock` is architecturally equivalent to PyTorch's `nn.TransformerEncoderLayer` with `norm_first=True`.
- Training from scratch on small data confirms the paper's observation that ViT lacks CNN inductive biases and requires large-scale pre-training to generalize.
- Feature extraction with pretrained `ViT-B/16` weights significantly outperforms the from-scratch baseline on the same small dataset, validating the paper's transfer learning findings.

---

## `Reference`

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

---

## `Acknowledgements`

Architecture design follows the original ViT paper by Dosovitskiy et al. (2021). Model variants and hyperparameters are sourced directly from Tables 1 and 3 of the paper. The `torchvision` pretrained weights correspond to the `ViT_B_16_Weights.DEFAULT` checkpoint.


