from flask import Flask, request, jsonify
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('SenAna.h5')

# Load the IMDb word index
word_index = imdb.get_word_index()

# Load test data and evaluate the model
(_, _), (X_test, y_test) = imdb.load_data(num_words=5000)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Function to preprocess text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Convert words to IDs using the IMDb word index
    tokens = [word_index.get(word, 0) for word in tokens if word_index.get(word, 0) < 5000]  # Filter out words not in top_words
    return tokens

# Route for predicting sentiment
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    review = request.json.get('review')
    if not review:
        return jsonify({'error': 'Review not provided'}), 400
    processed_review = preprocess_text(review)
    processed_review = pad_sequences([processed_review], maxlen=500)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    # Use actual accuracy obtained from model evaluation
    accuracy = scores[1] * 100
    return jsonify({'sentiment': sentiment, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
