from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from main import preprocess_input, predict_price  # Import your preprocess and predict functions

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')  # Make sure you have a 'model.pkl' file saved from your training

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Extract features from the JSON request
    try:
        features = np.array([
            data['area'],
            data['bedrooms'],
            data['bathrooms'],
            data['garage'],
            data['year_built']
        ]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400

    # Preprocess the input features
    features = preprocess_input(features)  # Assume this function preprocesses the data as needed

    # Make prediction
    try:
        prediction = predict_price(model, features)  # Assume this function predicts the price
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the prediction as JSON
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
