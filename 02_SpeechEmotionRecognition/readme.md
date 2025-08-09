# Audio Emotion Recognition Using CNN on TESS Toronto Dataset

This project implements an **audio emotion recognition** system using a Convolutional Neural Network (CNN). It uses Mel-Frequency Cepstral Coefficients (MFCC) features extracted from audio recordings of the **TESS Toronto Emotional Speech Set**.

---

## Dataset

- **TESS (Toronto emotional speech set)**:  
  A public dataset containing recordings of actors speaking various sentences with different emotional tones (happy, sad, angry, etc.).  
  [TESS dataset details](https://tspace.library.utoronto.ca/handle/1807/24487)

---

## Features

- Audio files are preprocessed by extracting MFCC features, which capture important characteristics of speech.
- MFCC features are padded or trimmed to a fixed length to maintain consistent input size for the CNN.
- A CNN model is trained to classify emotions based on these features.

---

## Model

- Convolutional Neural Network (CNN) architecture.
- Trained on MFCC features extracted from TESS dataset.
- Saves model weights as `emotion_cnn.h5`.
- Label encoder saved as `label_encoder.pkl` to map predictions to emotion labels.

---

## Usage

- Load the saved model and label encoder.
- Extract MFCC features from new audio input.
- Predict emotion using the trained CNN model.

---

## Deployment

- The model can be deployed via a web app using **Streamlit** or **Hugging Face Spaces**.
- Users can upload WAV audio files to get real-time emotion predictions.

---

## Requirements

- Python 3.7+
- TensorFlow
- Librosa
- NumPy
- Streamlit (for deployment)

---

## How to run locally

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
