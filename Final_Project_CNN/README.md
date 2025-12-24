# 🫁 Pneumonia Detection using CNN

A deep learning project that uses **Convolutional Neural Networks (CNN)** to detect pneumonia from chest X-ray images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project implements a CNN-based classifier to distinguish between **Normal** and **Pneumonia** cases from chest X-ray images. The model is trained on the [keremberke/chest-xray-classification](https://huggingface.co/datasets/keremberke/chest-xray-classification) dataset from HuggingFace.

## 🎯 Features

- Automatic dataset download from HuggingFace
- Custom CNN architecture optimized for medical imaging
- Data augmentation to prevent overfitting
- Weighted sampling for imbalanced classes
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, AUC-ROC)

## 📊 Dataset

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 4,077 | Training data |
| Validation | 1,165 | Model tuning |
| Test | 582 | Final evaluation |

**Classes:**
- `0` - NORMAL 🟢
- `1` - PNEUMONIA 🔴

## 🏗️ Model Architecture

```
Input (128x128x1)
    ↓
Conv Block 1: Conv2d(1→32) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 4: Conv2d(128→256) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Flatten (16,384)
    ↓
FC: 16384 → 256 → 128 → 1 (Sigmoid)
    ↓
Output: Probability [0, 1]
```

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 128×128 |
| Batch Size | 32 |
| Epochs | 25 (with early stopping) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |

## 🚀 Quick Start

### Installation

```bash
pip install datasets torch torchvision tqdm scikit-learn seaborn matplotlib pillow
```

### Training

Run the Jupyter notebook:
```bash
jupyter notebook pneumonia_detection_notebook_documented_CNN.ipynb
```

Or use Google Colab for GPU acceleration.

## 📈 Results

The model achieves competitive performance on the test set with:
- High sensitivity for pneumonia detection
- Balanced precision-recall trade-off
- Strong AUC-ROC score

## 🛡️ Techniques Used

- **Data Augmentation**: Random rotation, horizontal flip, affine transforms
- **Regularization**: Dropout (0.25 in conv, 0.5 in FC), Batch Normalization
- **Class Balancing**: Weighted random sampling
- **Optimization**: Learning rate scheduling with ReduceLROnPlateau
- **Early Stopping**: Patience of 5 epochs

## 📁 Project Structure

```
Final_Project_CNN/
├── pneumonia_detection_notebook_documented_CNN.ipynb  # Main notebook
├── README.md                                          # This file
└── README_Arabic.md                                   # Arabic documentation
```

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [CNN Explained - CS231n](https://cs231n.github.io/convolutional-networks/)

## 📄 License

This project is for educational purposes as part of a Neural Network and Deep Learning course.

---

**Made with ❤️ for medical AI research**
