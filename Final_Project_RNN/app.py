import numpy as np
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding

# Initialize FastAPI
app = FastAPI(
    title="Emoji Predictor API",
    description="Predict emoji from text using LSTM model",
    version="1.0.0"
)

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
word_to_index = None
word_to_vec_map = None
maxLen = 10

emoji_dictionary = {
    0: "❤️",
    1: "⚾",
    2: "😄",
    3: "😞",
    4: "🍴"
}

emoji_meanings = {
    0: "love",
    1: "sports",
    2: "happy",
    3: "sad",
    4: "food"
}

# Request/Response models
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    emoji: str
    emoji_meaning: str
    confidence: float
    all_predictions: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Helper functions
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        for w in sorted(words):
            words_to_index[w] = i
            i = i + 1
    return words_to_index, word_to_vec_map

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = (X[i].lower()).split()
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            j = j + 1
            if j >= max_len:
                break
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = 50
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        if word in word_to_vec_map:
            emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

def build_model(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)
    model = Model(sentence_indices, X)
    return model


@app.on_event("startup")
async def load_model():
    global model, word_to_index, word_to_vec_map
    
    print("Loading GloVe vectors...")
    word_to_index, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
    
    print("Building model...")
    model = build_model((maxLen,), word_to_vec_map, word_to_index)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Load weights if exists
    if os.path.exists('model.weights.h5'):
        print("Loading trained weights...")
        model.load_weights('model.weights.h5')
    else:
        print("Warning: No trained weights found. Model will use random weights.")
    
    print("Model loaded successfully!")

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="running",
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictResponse)
async def predict_emoji(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Prepare input
    x_test = np.array([text])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    
    # Predict
    predictions = model.predict(X_test_indices, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # All predictions with probabilities
    all_preds = {
        emoji_dictionary[i]: {
            "probability": float(predictions[0][i]),
            "meaning": emoji_meanings[i]
        }
        for i in range(5)
    }
    
    return PredictResponse(
        text=text,
        emoji=emoji_dictionary[predicted_class],
        emoji_meaning=emoji_meanings[predicted_class],
        confidence=confidence,
        all_predictions=all_preds
    )

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for text in texts:
        x_test = np.array([text.strip()])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
        predictions = model.predict(X_test_indices, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        
        results.append({
            "text": text,
            "emoji": emoji_dictionary[predicted_class],
            "emoji_meaning": emoji_meanings[predicted_class],
            "confidence": float(predictions[0][predicted_class])
        })
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
