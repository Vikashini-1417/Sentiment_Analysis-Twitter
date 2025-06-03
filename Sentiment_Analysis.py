import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# Load datasets
train_df = pd.read_csv('twitter_training.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Tweet'])
val_df = pd.read_csv('twitter_validation.csv', header=None, names=['ID', 'Entity', 'Sentiment', 'Tweet'])
# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])
# Apply cleaning
train_df['Cleaned_Tweet'] = train_df['Tweet'].apply(clean_text)
val_df['Cleaned_Tweet'] = val_df['Tweet'].apply(clean_text)
# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['Sentiment'])
y_val = le.transform(val_df['Sentiment'])
# Tokenization and sequence preparation
vocab_size = 10000
max_length = 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['Cleaned_Tweet'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['Cleaned_Tweet']), maxlen=max_length, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(val_df['Cleaned_Tweet']), maxlen=max_length, padding='post')

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Build LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict
)

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluate model
y_pred = np.argmax(model.predict(X_val), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))

def predict_sentiment(text):
    cleaned = clean_text(text)  # Clean the input like training data
    seq = tokenizer.texts_to_sequences([cleaned])  # Convert to sequence
    padded = pad_sequences(seq, maxlen=max_length, padding='post')  # Pad to max_length
    pred = model.predict(padded)
    label_index = np.argmax(pred)  # Get the index of highest probability
    sentiment = le.inverse_transform([label_index])[0]  # Convert index back to label
    return sentiment

# Ask user to enter a tweet
tweet = input("Enter a tweet to analyze sentiment: ")
result = predict_sentiment(tweet)
print("Sentiment:", result)


