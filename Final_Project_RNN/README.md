# 🎯 Emoji Predictor - Sentiment Analysis with LSTM

A deep learning project that uses **LSTM (Long Short-Term Memory)** networks to predict the appropriate emoji for any English sentence.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 📋 Overview

This project implements a sentiment analysis system that takes an English sentence as input and predicts the most suitable emoji. For example:
- Input: `"I love you"` → Output: ❤️
- Input: `"Let's play baseball"` → Output: ⚾
- Input: `"I am so happy"` → Output: 😄

## 🎯 Supported Emojis

| Class | Emoji | Meaning |
|-------|-------|---------|
| 0 | ❤️ | Love |
| 1 | ⚾ | Sports |
| 2 | 😄 | Happy |
| 3 | 😞 | Sad |
| 4 | 🍴 | Food |

## 🏗️ Model Architecture

```
Input: Sentence (max 10 words)
    ↓
Embedding Layer (GloVe 50d) - 400,001 × 50
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (50%)
    ↓
LSTM Layer 2 (128 units)
    ↓
Dropout (50%)
    ↓
Dense Layer (5 units, softmax)
    ↓
Output: Probability distribution over 5 emojis
```

**Total Parameters:** ~20.2M (mostly from embeddings)

## 🚀 Quick Start

### Option 1: Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download GloVe embeddings:**
   - Download from [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip)
   - Extract `glove.6B.50d.txt` to the project folder

3. **Train the model:**
```bash
python main.py
```

4. **Run the API:**
```bash
python app.py
```

### Option 2: Docker

```bash
docker build -t emoji-predictor .
docker run -p 7860:7860 emoji-predictor
```

### Option 3: Hugging Face Spaces

The model is deployed on Hugging Face Spaces. See `README_HuggingFace.md` for details.

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Single Sentence
```bash
POST /predict
Content-Type: application/json

{
    "text": "I love you"
}
```

**Response:**
```json
{
    "text": "I love you",
    "emoji": "❤️",
    "emoji_meaning": "love",
    "confidence": 0.95,
    "all_predictions": {...}
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

["I love you", "Let's eat pizza", "I am sad"]
```

## 📁 Project Structure

```
Final_Project_RNN/
├── main.py              # Training script
├── app.py               # FastAPI server
├── emo_utils.py         # Helper functions
├── train_emoji.csv      # Training data
├── test_emoji.csv       # Test data
├── glove.6B.50d.txt     # Word embeddings (download separately)
├── model.weights.h5     # Trained model weights
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── README.md            # This file
└── README_Arabic.md     # Arabic documentation
```

## 📊 Training Results

```
Training Accuracy: ~98.5%
Test Accuracy: ~87.5%
```

## 🧠 Key Concepts

### Why LSTM?
- Handles sequential data (words in a sentence)
- Solves the vanishing gradient problem of vanilla RNNs
- Maintains long-term dependencies through cell state

### Why GloVe Embeddings?
- Pre-trained word vectors capture semantic meaning
- Similar words have similar vectors
- Reduces training data requirements

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Max Sentence Length | 10 words |
| Embedding Dimension | 50 |
| LSTM Hidden Units | 128 |
| Dropout Rate | 0.5 |
| Epochs | 100 |
| Batch Size | 32 |
| Optimizer | Adam |

## 📚 References

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Keras Documentation](https://keras.io/)

## 📄 License

This project is for educational purposes as part of a Neural Network and Deep Learning course.

---

**Made with ❤️ using LSTM and GloVe**
