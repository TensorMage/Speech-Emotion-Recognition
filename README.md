# 🎙️ Speech Emotion Recognition: Traditional ML to Deep Learning

A comprehensive Speech Emotion Recognition (SER) project that explores the complete journey from **traditional Machine Learning** to **Deep Learning** using Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs).

Instead of implementing a single model, this project investigates how different approaches perform on the same problem while highlighting the trade-offs between handcrafted audio features and end-to-end feature learning.

---

# 📖 Project Overview

Speech Emotion Recognition aims to identify the emotional state of a speaker from an audio recording.

This project was developed in multiple stages:

1. **Traditional Machine Learning**
   - Audio preprocessing
   - Handcrafted feature extraction
   - SVM classifier

2. **Artificial Neural Networks (ANN)**
   - Using extracted acoustic features
   - Built completely in PyTorch
   - GPU accelerated training

3. **Convolutional Neural Networks (CNN)**
   - Raw audio converted into Mel Spectrograms
   - Spectrogram Dataset generation
   - End-to-end image classification using CNNs

The objective was not only to maximize accuracy but also to understand the strengths and limitations of different approaches for Speech Emotion Recognition.

---

# 🚀 Project Evolution

## Stage 1 — Traditional Machine Learning

```
Audio (.wav)
      │
      ▼
Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Feature Scaling
      │
      ▼
Support Vector Machine
      │
      ▼
Emotion Prediction
```

### Extracted Features

- MFCC
- Chroma
- RMS Energy
- Zero Crossing Rate
- Spectral Centroid
- Spectral Contrast
- Spectral Rolloff

These handcrafted acoustic features were used to train a Support Vector Machine classifier.

**Accuracy:** ~69%

---

## Stage 2 — Artificial Neural Network (ANN)

Instead of a traditional classifier, the extracted feature vectors were used to train an Artificial Neural Network implemented completely in PyTorch.

### ANN Pipeline

```
Extracted Features
        │
        ▼
StandardScaler
        │
        ▼
TensorDataset
        │
        ▼
DataLoader
        │
        ▼
Artificial Neural Network
        │
        ▼
Emotion Prediction
```

### Concepts Implemented

- PyTorch tensors
- TensorDataset
- DataLoader
- GPU training
- Forward propagation
- Backpropagation
- Batch Normalization
- Dropout
- Adam Optimizer
- CrossEntropyLoss

**Best Accuracy:** ~75%

---

## Stage 3 — Convolutional Neural Network (CNN)

Instead of using handcrafted features, the CNN learns directly from Mel Spectrograms generated from the raw speech recordings.

### CNN Pipeline

```
Audio (.wav)
      │
      ▼
Mel Spectrogram
      │
      ▼
Normalization
      │
      ▼
128×128 Spectrogram
      │
      ▼
Convolutional Neural Network
      │
      ▼
Emotion Prediction
```

### CNN Architecture

- 3 Convolution Blocks
- Batch Normalization
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Dropout
- Adam Optimizer

### Data Pipeline

Instead of generating spectrograms every epoch:

```
.wav
    │
    ▼
Generate Mel Spectrogram
    │
    ▼
Save as .npy
    │
    ▼
PyTorch Dataset
    │
    ▼
GPU Training
```

This significantly reduced preprocessing overhead during training.

Current CNN Accuracy: **~69%**

Although the CNN achieved high training accuracy, experiments revealed strong overfitting due to the limited size of the RAVDESS dataset.

---

# 📊 Model Comparison

| Model | Input | Accuracy |
|--------|------|----------|
| Support Vector Machine | Handcrafted Features | **69%** |
| Artificial Neural Network | Handcrafted Features | **75%** |
| Convolutional Neural Network | Mel Spectrograms | **69%** |

---

# 📁 Project Structure

```
Speech-Emotion-Recognition/
│
├── ANN/
│   ├── ann_training.ipynb
│   └── trained_ann_model.pth
│
├── CNN/
│   ├── cnn_training.ipynb
│   ├── spectrogram_generation.ipynb
│   └── trained_cnn_model.pth
│
├── Traditional ML/
│   ├── svm_training.ipynb
│   └── emotion_model.pkl
│
├── app.py
├── predict.py
├── feature_extraction.py
├── speech_features.csv
├── requirements.txt
└── README.md
```

---

# 📂 Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

It contains professionally recorded speech samples representing eight different emotions.

---

# 🛠 Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-learn

### Deep Learning

- PyTorch

### Audio Processing

- Librosa

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib

### Deployment

- Streamlit

---

# 🧠 What I Learned

This project was designed as a learning journey through Deep Learning.

### PyTorch Fundamentals

- Custom Dataset
- DataLoader
- GPU Training
- Training Loop
- Evaluation Loop
- Saving and Loading Models

### Neural Networks

- Forward Propagation
- Backpropagation
- Gradient Descent
- CrossEntropyLoss
- Adam Optimizer

### CNN Concepts

- Convolution
- Feature Maps
- Padding
- Stride
- Pooling
- Batch Normalization
- Dropout
- Mel Spectrograms
- End-to-End Learning

### Speech Processing

- Audio Preprocessing
- Spectrogram Generation
- Feature Engineering
- Emotion Classification

---

# 🔬 Key Observations

One of the primary objectives of this project was to compare handcrafted feature-based learning with end-to-end deep learning.

Some important observations:

- Handcrafted acoustic features remain highly effective for small speech datasets.
- ANNs generalized better than a CNN trained from scratch on the RAVDESS dataset.
- CNNs demonstrated strong learning capability but showed noticeable overfitting due to the limited number of training samples.
- The project highlighted the importance of validation strategies, data augmentation, and larger datasets for deep learning models.

---

# 🚧 Future Work

The current CNN serves as a baseline for future improvements.

Planned enhancements include:

- Transfer Learning using ResNet18 / EfficientNet
- SpecAugment
- Time & Frequency Masking
- Audio Data Augmentation
- Early Stopping
- Learning Rate Scheduling
- Speaker-independent Evaluation
- Multi-dataset Training (RAVDESS + TESS + CREMA-D + SAVEE)
- Vision Transformers / Audio Spectrogram Transformer (AST)

## ⭐ Project Goal

This project is not only about building a Speech Emotion Recognition system but also about understanding **how different Machine Learning and Deep Learning approaches behave on the same problem**, analyzing their strengths, weaknesses, and generalization capabilities.
