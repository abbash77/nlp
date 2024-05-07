import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Download NLTK resources
download('punkt')
download('stopwords')

# Load the IMDb dataset using pandas
imdb_data = pd.read_csv("IMDB-Dataset.csv")

# Define stopwords and preprocess function
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Preprocess text data
imdb_data['tokens'] = imdb_data['review'].apply(preprocess_text)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=imdb_data['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Get word embeddings
word_embeddings = word2vec_model.wv

# Pad sequences to ensure uniform length
max_len = max(imdb_data['tokens'].apply(len))
X = imdb_data['tokens'].apply(lambda tokens: [word_embeddings.key_to_index[word] for word in tokens if word in word_embeddings.key_to_index])
X = pad_sequences(X, maxlen=max_len, padding='post')

# Convert sentiment labels to categorical
y = to_categorical((imdb_data['sentiment'] == 'positive').astype(int))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model architecture
model = Sequential([
    Embedding(input_dim=len(word_embeddings.key_to_index)+1, output_dim=100, input_length=max_len),
    LSTM(units=128),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")