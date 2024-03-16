import React, { useState } from 'react';
import './App.css';
import bgSrc from "./assets/symbol.jpg";
import axios from 'axios'; // Import Axios


const API_URL = 'http://localhost:5000/predict_sentiment'; // Replace with your actual sentiment analysis API endpoint

const App = () => {
  const [inputText, setInputText] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [accuracy, setAccuracy] = useState('');
  const [emotion, setEmotion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null); // Reset error state

    try {
      // const response = await fetch(API_URL, {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      //   body: JSON.stringify({ review: inputText }),
      // });
      const formData = new FormData();
      formData.append('review', inputText);

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status !== 200) {
        throw new Error('Failed to analyze sentiment');
      }

      // const result = await response.json();
      const result = await response.data;
      console.log(result)
      setSentiment(result.sentiment);
      setAccuracy(result.accuracy);
      setEmotion(result.emotion);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      setError('An error occurred while analyzing sentiment. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container ">
        <div className="row mb-5">
          <div className="col-md-6 col-sm-12">
            <div className="bg-grey text-white mb-4">
              <h2 className="title display-1 font-weight-bold text-center mt-5">Sentiment Analysis</h2>
              <h4 className="font-weight-bold mt-5">Uncover Emotions, Harness Insights: Movie Reviews Analyzed, Empowering Better Understanding.</h4>
            </div>
          </div>
          <div className="col-md-6 col-sm-12">
            <div className="d-flex justify-content-end">
              <img className="symbol img-fluid mt-5" src={bgSrc} alt="Background scenery" />
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-6 col-sm-12 justify-content-start">
            <label className='d-flex mt-3'>Enter the review : </label>
            <textarea className='d-flex mt-3'
              placeholder="Enter text for sentiment analysis"
              value={inputText}
              onChange={handleInputChange}
            />
            {error && <div className="error">{error}</div>}
            <button className='d-flex mt-3' onClick={handleSubmit} disabled={loading}>
              {loading ? 'Analyzing...' : 'Submit'}
            </button>
          </div>
          <div className="col-md-6 col-sm-12">
            <div className="result">
              <label>Sentiment:</label>
              <span>{sentiment}</span>
            </div>
            <div className="result">
              <label>Accuracy:</label>
              <span>{accuracy}</span>
            </div>
            <div className="result">
          <label>Emotion:</label>
          <span>{emotion}</span>
        </div> 
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
