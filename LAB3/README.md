# IT549: Deep Learning – Lab 3
## Image-Based AQI Classification using CNN and Pretrained Models

---

## Project Overview

This project builds an image classification pipeline to predict Air Quality Index (AQI) categories from location images using two deep learning approaches:

1. **BasicCNN** – Trained from scratch
2. **ResNet-18** – Transfer learning with first 10 of 18 layers frozen

---

## Dataset

- **Source:** [Google Drive Dataset](https://drive.google.com/drive/folders/1usBxgNB67GfhCQ2f7xRkDlF6fgIZZrP?usp=sharing)
- **`data.csv`** – Maps image filenames to AQI_Class labels
- **`sampled_images/`** – 6,000 images, 6 balanced classes (1,000 each)

### AQI Classes
| Index | Class |
|-------|-------|
| 0 | a_Good |
| 1 | b_Moderate |
| 2 | c_Unhealthy_for_Sensitive_Groups |
| 3 | d_Unhealthy |
| 4 | e_Very_Unhealthy |
| 5 | f_Severe |

---

## Project Structure

```
├── IT549_Lab3_AQI_Classification.ipynb   # Main Colab notebook
├── data.csv
├── sampled_images/
├── results/
│   ├── training_curves.png
│   ├── cm_cnn.png
│   ├── cm_resnet18.png
│   ├── model_comparison.png
│   ├── misclassified_cnn.png
│   ├── misclassified_resnet18.png
│   ├── results_summary.csv
│   ├── BasicCNN_best.pt
│   └── ResNet18_TL_best.pt
└── README.md
```

---

## Setup & Installation

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow
```

---

## How to Run

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Upload dataset to Google Drive and update the path variables:
```python
CSV_PATH = "/content/drive/MyDrive/IT549_Lab3/data.csv"
IMG_DIR  = "/content/drive/MyDrive/IT549_Lab3/sampled_images"
```
3. Set runtime to **GPU** → Runtime → Change runtime type → T4 GPU
4. Run all cells

---

## Model Architectures

### BasicCNN (from scratch)
```
Conv(3→32) → BN → ReLU → MaxPool
Conv(32→64) → BN → ReLU → MaxPool
Conv(64→128) → BN → ReLU → MaxPool
Conv(128→256) → BN → ReLU → MaxPool
GlobalAvgPool → FC(256) → ReLU → Dropout(0.5) → FC(6)
```

### ResNet-18 (Transfer Learning)
```
[FROZEN  – layers 00–09]  conv1, bn1, layer1, layer2
[TRAINED – layers 10–17]  layer3, layer4
[NEW HEAD]  Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→6)
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Learning rate | 0.001 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Frozen layers (ResNet-18) | 10 / 18 |
| Random seed | 42 |
| Train / Val / Test split | 70 / 15 / 15 |

---

## Results

### Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| BasicCNN (scratch) | 0.7744 | 0.7729 | 0.7744 | 0.7672 |
| ResNet-18 (TL, freeze=10) | 0.7689 | 0.7726 | 0.7689 | 0.7681 |

### Per-Class F1-Score

| Class | BasicCNN | ResNet-18 TL |
|-------|----------|--------------|
| a_Good | 0.82 | 0.72 |
| b_Moderate | 0.75 | 0.72 |
| c_Unhealthy_for_Sensitive_Groups | 0.54 | 0.66 |
| d_Unhealthy | 0.74 | 0.72 |
| e_Very_Unhealthy | 0.85 | 0.88 |
| f_Severe | 0.90 | 0.91 |

---

## Discussion

### Why both models perform similarly (~77%)
The dataset contains visually subtle differences between adjacent AQI classes (e.g. Good vs Moderate, Unhealthy vs Very Unhealthy), making classification inherently difficult from image features alone.

### Where BasicCNN struggles
`c_Unhealthy_for_Sensitive_Groups` has the lowest F1 (0.54) — the class sits between two visually similar neighbours and the model trained from scratch lacks the feature depth to separate them.

### Where ResNet-18 TL helps
ResNet-18 improves precisely on the hard middle class (`c_Unhealthy_for_Sensitive_Groups`: 0.54 → 0.66) thanks to richer pretrained features, confirming that transfer learning helps where visual discrimination is most subtle.

### Why transfer learning didn't dominate overall
- Only 10 of 18 layers were frozen — the 8 trainable layers still needed sufficient data to adapt well
- AQI scenes differ significantly from ImageNet object categories, so the ImageNet prior helps less than it would on natural object classification
- Both models converge to similar overall accuracy because the bottleneck is class separability in the images, not model capacity
