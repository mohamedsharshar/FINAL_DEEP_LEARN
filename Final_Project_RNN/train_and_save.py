"""
Script to train the model and save weights for deployment
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding
import csv

np.random.seed(1)

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

def read_csv(filename):
    phrase = []
    emoji = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])
    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)
    return X, Y

def convert_to_one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]

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

if __name__ == "__main__":
    print("Loading data...")
    X_train, Y_train = read_csv('train_emoji.csv')
    X_test, Y_test = read_csv('test_emoji.csv')
    maxLen = 10
    
    print("Loading GloVe vectors...")
    word_to_index, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
    
    print("Building model...")
    model = build_model((maxLen,), word_to_vec_map, word_to_index)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining model...")
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)
    model.fit(X_train_indices, Y_train_oh, epochs=100, batch_size=32, shuffle=True)
    
    print("\nEvaluating model...")
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print(f"Test accuracy: {acc:.4f}")
    
    print("\nSaving model weights...")
    model.save_weights('model.weights.h5')
    print("Model weights saved to 'model.weights.h5'")
    
    print("\nTesting prediction...")
    test_text = "I love you"
    x = np.array([test_text])
    x_indices = sentences_to_indices(x, word_to_index, maxLen)
    pred = model.predict(x_indices)
    emoji_dict = {0: "❤️", 1: "⚾", 2: "😄", 3: "😞", 4: "🍴"}
    print(f"'{test_text}' -> {emoji_dict[np.argmax(pred)]}")
    
    print("\n✅ Done! Files ready for deployment:")
    print("   - model.weights.h5")
    print("   - app.py")
    print("   - requirements.txt")
    print("   - glove.6B.50d.txt")
