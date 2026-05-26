# Weather Image Classification using Vision Transformer (ViT)

---

## Overview

This project implements a Vision Transformer (ViT-Base/16) model to classify weather conditions from photographs across 11 fine-grained categories. The model is fine-tuned from pretrained ImageNet weights and achieves **92.82% test accuracy** with a macro F1-score of 0.93.

---

## Weather Categories

| Class | Class | Class |
|-------|-------|-------|
| Dew | Fogsmog | Frost |
| Glaze | Hail | Lightning |
| Rain | Rainbow | Rime |
| Sandstorm | Snow | |

---

## Model Architecture

- **Backbone:** ViT-Base/16 (pretrained on ImageNet-21k)
- **Input Resolution:** 224 × 224 pixels
- **Patch Size:** 16 × 16 (196 patches per image)
- **Transformer Layers:** 12
- **Attention Heads:** 12
- **Embedding Dimension:** 768
- **Total Parameters:** ~85.8 Million
- **Classification Head:** Linear layer → 11 classes

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92.82% |
| Macro Precision | 0.93 |
| Macro Recall | 0.93 |
| Macro F1-Score | 0.93 |
| Best Epoch | 7 |
| Total Epochs Trained | 12 |

### Per-Class F1 Scores

| Class | F1-Score |
|-------|----------|
| Rainbow | 1.00 ⭐ |
| Lightning | 0.99 |
| Hail | 0.99 |
| Dew | 0.98 |
| Sandstorm | 0.96 |
| Fogsmog | 0.94 |
| Rain | 0.93 |
| Rime | 0.92 |
| Frost | 0.86 |
| Snow | 0.85 |
| Glaze | 0.82 |

---

## Dataset

- **Total Images:** ~6,862
- **Split:** 70% Train / 15% Validation / 15% Test
- **Class Imbalance:** Mild (5x ratio between largest and smallest class)
- **Imbalance Handling:** Online data augmentation during training

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | StepLR (step=5, gamma=0.5) |
| Early Stopping Patience | 5 epochs |
| Max Epochs | 30 |

---

## Augmentations (Training Only)

- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation)

---

## Project Structure

```
Deep-Learning-Project-D/
│
├── README.md
├── requirements.txt
│
├── Group_XX_Notebook.ipynb
├── Group_XX_Report_Final.docx
│
├── checkpoints/
│   └── project_d_model.pt
│
└── figures/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── attention_maps.png
    ├── inference_demo.png
    └── failure_cases.png
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Ahmed-Saeed-Umar/Deep-Learning-Project-D.git
cd Deep-Learning-Project-D
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Google Colab (Recommended)
- Upload the notebook to [Google Colab](https://colab.research.google.com)
- Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
- Mount your Google Drive and update the `dataset_path` variable
- Run all cells

### 4. Load Pretrained Model
```python
import timm
import torch

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=11)
model.load_state_dict(torch.load('checkpoints/project_d_model.pt'))
model.eval()
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
opencv-python>=4.7.0
Pillow>=9.5.0
albumentations>=1.3.0
```

---

## Key Findings

- Pretrained ViT converges rapidly — strong results achieved within 7 epochs
- Classes with distinctive visual signatures (rainbow, lightning, hail) achieve near-perfect classification
- Main challenge is inter-class confusion among ice-related categories (frost, glaze, rime, snow) due to overlapping visual characteristics
- Attention maps confirm the model focuses on semantically meaningful image regions

---

## References

1. Dosovitskiy et al. — *An Image is Worth 16x16 Words* (ICLR 2021)
2. Vaswani et al. — *Attention is All You Need* (NeurIPS 2017)
3. Liu et al. — *Swin Transformer* (ICCV 2021)
