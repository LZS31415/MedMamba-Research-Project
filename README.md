# MedMamba Research Project

Reproducibility study, critical assessment, and alternative model comparison of **MedMamba: Vision Mamba for Medical Image Classification** (arXiv:2403.03849), evaluated on the PathMNIST dataset. Built for the MSI5102 group project.

**Team:** Arora Anuj, Chandiramani Lavnit Sanjeev, Lau Zhan Siang, Lim Feng Yueh, Liau Yun Qing, Rajagopalan Dayanithy

## What this project does

The work is split into three parts, each covering a distinct task we were assigned:

1. **Reproduce** the paper's reported results on PathMNIST using the authors' own published weights (`notebooks/part1_reproducibility.ipynb`).
2. **Challenge** those results by training the paper's own baseline models (ConvNeXt-T, ResNet50) under fairer conditions, to test whether MedMamba's advantage is architectural or a byproduct of how the baselines were trained (`notebooks/part2_alternative_models_challenge.ipynb`).
3. **Present** the findings (`slides/MedMamba_Presentation.pptx`).

## Repo structure

```
MedMamba-Research-Project/
├── README.md
├── notebooks/
│   ├── part1_reproducibility.ipynb              # Reproduces paper's MedMamba-T results on PathMNIST
│   └── part2_alternative_models_challenge.ipynb  # ConvNeXt-T / ResNet50 trained fairly, vs. paper baselines
└── slides/
    └── MedMamba_Presentation.pptx                # Class presentation
```

## Part 1: Reproducibility results

Evaluated MedMamba-T (14.5M params) on the PathMNIST test set (7,180 images, 9 classes) using the official pretrained weights.

| Metric | Paper | Ours | Difference |
|---|---|---|---|
| Overall Accuracy | 95.30% | 95.25% | 0.05% |
| AUC | 0.9970 | 0.9975 | 0.0005 |

**Verdict:** reproducibility confirmed, within expected variation.

**Key finding during reproduction:** the authors' training script uses PyTorch's `ImageFolder`, which assigns class indices alphabetically. MedMNIST uses a different, predefined class ordering. Without remapping labels to match, evaluation against MedMNIST ground truth is silently wrong. This mismatch is undocumented in both the paper and the original repository.

## Part 2: Alternative model comparison

The paper's own baselines (ConvNeXt-T, ResNet50) were trained from scratch, with no pretrained weights and no data augmentation, while MedMamba is the paper's proposed architecture. We re-trained the two baselines under standard, fair conditions (ImageNet pretrained weights, AutoAugment, same optimizer/schedule) to test whether MedMamba's reported gains are architectural or a result of an uneven playing field.

| Model | Training conditions | OA (%) | AUC |
|---|---|---|---|
| MedMamba-T (paper) | no pretrain, no augmentation | 95.30 | 0.9970 |
| MedMamba-S (paper) | no pretrain, no augmentation | 95.50 | 0.9970 |
| ResNet50 (paper) | no pretrain, no augmentation | 89.20 | 0.9890 |
| ConvNeXt-T (ours) | ImageNet pretrained + AutoAugment | **97.59** | 0.9978 |
| ResNet50 (ours) | ImageNet pretrained + AutoAugment | **96.28** | 0.9981 |

**Finding:** when trained fairly, both ConvNeXt-T and ResNet50 outperform every MedMamba variant reported in the paper. This doesn't mean MedMamba is a bad architecture, it is genuinely efficient and performs well. It means the paper's baseline comparisons were run under conditions that suppressed the baselines' true performance, so the reported margin over CNNs overstates MedMamba's real-world advantage on this dataset.

## Setup and how to run

Both notebooks are designed for Google Colab with a T4 GPU.

1. Create a folder in your Google Drive for this project.
2. Download the official pretrained weights (`Medmamba.pth`) from the [MedMamba repo](https://github.com/YubiaoYue/MedMamba) and place them in that folder.
3. Download the PathMNIST dataset (`pathmnist_224.npz`, ~12.6 GB) from [Zenodo](https://zenodo.org/records/10519652) and place it in the same folder.
4. Open `notebooks/part1_reproducibility.ipynb` in Colab, set the runtime to a T4 GPU (Runtime → Change runtime type → GPU), and update the paths at the top of the notebook to point at your Drive folder.
5. Run all cells top to bottom:
   - The PyTorch/CUDA reinstall step requires a runtime restart when it finishes, then continue from the next step.
   - The `mamba-ssm` install step builds from source the first time (~25 minutes) and caches the wheels to a `mamba_wheels_cache` folder in your Drive. Every run after that installs from the cache (~2 minutes). **Don't delete that cache folder**, or you'll pay the 25-minute build again.
   - The full evaluation step takes 5-10 minutes; let it finish without interrupting.
6. If the notebook prints the reproducibility summary at the end, the run succeeded.
7. `notebooks/part2_alternative_models_challenge.ipynb` uses the same PathMNIST file and only needs `torchvision`, no `mamba-ssm` install required. Training both models takes several hours on a T4 (see per-epoch timing in the notebook output).

## Reference

This project reproduces and critiques:

> Yue, Y., & Li, Z. (2024). *MedMamba: Vision Mamba for Medical Image Classification.* arXiv:2403.03849. https://arxiv.org/abs/2403.03849

Original code and pretrained weights: https://github.com/YubiaoYue/MedMamba

This repository contains our own reproduction, evaluation, and analysis code. It does not redistribute the original paper PDF or the authors' pretrained weights; both are available at the links above.
