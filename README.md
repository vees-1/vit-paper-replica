# Vision Transformer (ViT)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-0.25-orange)
![Device](https://img.shields.io/badge/Device-Apple%20MPS-black?logo=apple&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.58%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A from-scratch implementation of the **Vision Transformer (ViT)** architecture based on the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), applied to a 3-class food classification task.

---

## Overview

- Built and trained a ViT model from scratch in PyTorch
- Fine-tuned a pretrained ViT (`vit_b_16`) as a feature extractor
- Device: Apple MPS (Metal Performance Shaders)

## Results 
(Pretrained `ViT_b_16`)

| Metric | Value |
|---|---|
| Train Loss | 0.0658 |
| Train Accuracy | 98.33% |
| Test Loss | 0.0633 |
| Test Accuracy | 98.58% |

Minimal generalization gap indicates stable training with effective regularization.

## Usage

1. Clone the repo and install dependencies.
2. Run `vit.ipynb` — data will be downloaded automatically if not present.
3. The trained model is saved to `models/` at the end of the notebook.

## References

- [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929)
