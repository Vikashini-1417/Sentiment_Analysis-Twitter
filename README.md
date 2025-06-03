# Twitter Sentiment Analysis using LSTM

This project performs sentiment analysis on Twitter data using a Bidirectional LSTM model. It classifies tweets into sentiments like `Positive`, `Negative`, `Neutral`, or others (based on dataset labels).

## 📁 Dataset

The model is trained on two CSV files:
- `twitter_training.csv`
- `twitter_validation.csv`

Each file is expected to have the following columns:
1. **ID** – Unique identifier
2. **Entity** – Subject or target entity
3. **Sentiment** – The sentiment label (`Positive`, `Negative`, `Neutral`, etc.)
4. **Tweet** – The tweet text

## ⚙️ Requirements

- Python 3.7+
- TensorFlow
- NumPy
- pandas
- scikit-learn
- NLTK
- matplotlib

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn nltk matplotlib
Also, download NLTK stopwords if not already installed:

python

import nltk
nltk.download('stopwords')
🧹 Preprocessing
Remove URLs, mentions, hashtags, punctuation, and numbers

Convert to lowercase

Remove stopwords using NLTK

🧠 Model
A Bidirectional LSTM with:

Embedding Layer

LSTM Layer (64 units)

Dropout (0.3)

Dense hidden layer (32 units)

Output Layer with softmax activation for multi-class classification

🏋️ Training
Tokenized and padded tweet sequences

Trained for 10 epochs with sparse_categorical_crossentropy loss

Handled class imbalance using class_weight

📈 Visualization
The script plots:

Training and validation accuracy

Training and validation loss

📊 Evaluation
After training, the model prints a full classification report including precision, recall, and F1-score.

🔮 Prediction
You can input any tweet and the model will predict its sentiment:
Enter a tweet to analyze sentiment: I love this new phone!
Sentiment: Positive

📂 File Structure

.
├── twitter_training.csv
├── twitter_validation.csv
├── sentiment_analysis_lstm.py
└── README.md
