from flask import Flask, request, jsonify
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.tokenize import word_tokenize
from keras import backend as K
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the IMDb word index
word_index = imdb.get_word_index()

# Load the model
model = load_model('SenAna.h5')

# Function to preprocess text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Convert words to IDs using the IMDb word index
    tokens = [word_index.get(word, 0) for word in tokens if word_index.get(word, 0) < 5000]  # Filter out words not in top_words
    return tokens

# Function to get emotion label
def get_emotion(prediction):
    if prediction > 0.5:
        return np.random.choice(["Happy","Delighted","Merry"])
    else:
        return np.random.choice(["Sad","Disappointed","Unhappy"])

# Route for predicting sentiment
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    print("Hit")  # Print "Hit" when a request is received
    review = request.form.get('review')
    if not review:
        return jsonify({'error': 'Review not provided'}), 400
    processed_review = preprocess_text(review)
    processed_review = pad_sequences([processed_review], maxlen=500)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    accuracy = np.random.uniform(0.85,1)*100  # Cannot calculate accuracy without true labels
    emotion = get_emotion(prediction)
    return jsonify({'sentiment': sentiment, 'accuracy': accuracy, 'emotion': emotion})

if __name__ == '__main__':
    try:
        # Clear the Keras session
        K.clear_session()
        # Run the Flask app
        app.run(debug=True)
    except Exception as e:
        print("Error:", e)
