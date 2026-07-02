# Speech Emotion Recognition

A Machine Learning project that predicts the emotion expressed in a speech recording using audio signal processing and traditional machine learning techniques. The project extracts meaningful audio features using Librosa, preprocesses them, and predicts the speaker's emotion through a trained classification model. A Streamlit web application is included for easy interaction.

---

## Overview

This project allows users to upload a speech recording in `.wav` format and predicts the emotion expressed by the speaker. The system performs audio preprocessing, extracts relevant speech features, scales the extracted features, and uses a trained machine learning model for prediction.

---

## Features

- Predicts emotions from speech recordings.
- Automatic audio preprocessing.
- Feature extraction using Librosa.
- Displays prediction confidence.
- Interactive Streamlit web application.

---

## Project Structure

```
Speech-Emotion-Recognition/
│
├── app.py                     # Streamlit application
├── predict.py                 # Prediction pipeline
├── feature_extraction.py      # Audio feature extraction
├── emotion_model.pkl          # Trained machine learning model
├── scaler.pkl                 # StandardScaler
├── label_encoder.pkl          # Label encoder
├── speech_features.csv        # Extracted feature dataset
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Scikit-learn
- Librosa
- NumPy
- Joblib
- Streamlit

---

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The dataset contains professionally recorded speech samples representing multiple emotional states and is widely used for Speech Emotion Recognition research.

---

## Audio Preprocessing

Each audio recording undergoes the following preprocessing steps before feature extraction:

- Silence trimming
- Audio normalization

These steps improve feature consistency and reduce unnecessary noise.

---

## Feature Extraction

Audio features are extracted using **Librosa**.

The extracted features include:

- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma Features
- Root Mean Square (RMS) Energy
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Spectral Contrast
- Spectral Rolloff

For every feature, both the **mean** and **standard deviation** are computed to generate a fixed-length feature vector suitable for machine learning models.

---

## Prediction Pipeline

```
Audio File
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
Emotion Prediction
      │
      ▼
Confidence Score
```

The trained model, feature scaler, and label encoder are loaded using Joblib during inference.

---

## Web Application

The project includes a Streamlit application that enables users to:

- Upload a WAV audio file
- Listen to the uploaded audio
- Predict the speaker's emotion
- Display the prediction confidence

Run the application using:

```bash
streamlit run app.py
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
```

Navigate to the project directory

```bash
cd Speech-Emotion-Recognition
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Requirements

The project depends on:

- Python
- Scikit-learn
- Librosa
- NumPy
- Joblib
- Streamlit

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

Possible future enhancements include:

- Support for additional speech datasets
- Further model optimization and evaluation
- Real-time microphone input
- Cloud deployment
- Emotion visualization dashboard

---

## Acknowledgements

- Librosa
- Scikit-learn
- Streamlit
- RAVDESS Dataset
