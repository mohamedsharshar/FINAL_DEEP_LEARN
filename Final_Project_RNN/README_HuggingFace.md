---
title: Emoji Predictor API
emoji: 😄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🎯 Emoji Predictor API

Predict the appropriate emoji for any English sentence using LSTM deep learning model.

## 🚀 API Endpoints

### Health Check
```
GET /
GET /health
```

### Predict Single Text
```
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
    "all_predictions": {
        "❤️": {"probability": 0.95, "meaning": "love"},
        "⚾": {"probability": 0.01, "meaning": "sports"},
        "😄": {"probability": 0.02, "meaning": "happy"},
        "😞": {"probability": 0.01, "meaning": "sad"},
        "🍴": {"probability": 0.01, "meaning": "food"}
    }
}
```

### Predict Batch
```
POST /predict/batch
Content-Type: application/json

["I love you", "I am hungry", "Let's play baseball"]
```

## 📊 Supported Emojis

| Emoji | Meaning |
|-------|---------|
| ❤️ | Love |
| ⚾ | Sports |
| 😄 | Happy |
| 😞 | Sad |
| 🍴 | Food |

## 🔧 Technology Stack

- FastAPI
- TensorFlow/Keras
- LSTM Neural Network
- GloVe Word Embeddings
