# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')

# Load the data
data = pd.read_csv('Emails.csv')
print("Data shape:", data.shape)

# Check class distribution
sns.countplot(x='spam', data=data)
plt.title('Spam and Non Spam Distribution')
plt.show()

# Balancing the data by downsampling
spamable_data = data[data.spam == 1]
non_spam_data = data[data.spam == 0].sample(len(spamable_data), random_state=88) # ham = non spam
balanced_data = pd.concat([spamable_data, non_spam_data], ignore_index=True)

# Preprocess the text data
def preprocess_text1(text):
    text = text.replace("Subject", "")
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return " ".join(word.lower() for word in text.split() if word.lower() not in stop_words)

balanced_data['text'] = balanced_data['text'].apply(preprocess_text1)

# Split the data into training, validation, and testing sets
train_X, temp_X, train_Y, temp_Y = train_test_split(
    balanced_data['text'], balanced_data['spam'], test_size=0.2, random_state=42
)
val_X, test_X, val_Y, test_Y = train_test_split(
    temp_X, temp_Y, test_size=0.5, random_state=42
)

# Tokenize and pad the text sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=100)
val_sequences = pad_sequences(tokenizer.texts_to_sequences(val_X), maxlen=100)
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=100)

# Set embedding dimensions
embedding_dim = 64

# Build the model with a trainable embedding layer
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

# Model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_sequences, train_Y, validation_data=(val_sequences, val_Y),
    epochs=15, batch_size=32, callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Generate predictions
predictions = (model.predict(test_sequences) > 0.5).astype("int32")

# Classification report
print(classification_report(test_Y, predictions, target_names=['Ham', 'Spam']))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
