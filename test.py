import unittest
import json
from app import app
from main import load_and_preprocess_data, train_model, predict_price

class TestHousePricePrediction(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # Test API endpoint for valid prediction
    def test_predict_valid(self):
        response = self.client.post('/predict', json={
            'features': [3000, 3, 2]  # Example feature inputs (area, bedrooms, bathrooms)
        })
        data = json.loads(response.data)
        self.assertIn('prediction', data)

    # Test API endpoint for missing feature data
    def test_predict_invalid(self):
        response = self.client.post('/predict', json={})
        data = json.loads(response.data)
        self.assertIn('error', data)

    # Test model training and prediction function
    def test_model_prediction(self):
        # Use a small subset of the data for quick testing
        X, y, preprocessor = load_and_preprocess_data('house_prices.csv')
        model = train_model(X, y)
        features = [3000, 3, 2]
        prediction = predict_price(features, model, preprocessor)
        self.assertIsInstance(prediction, float)

if __name__ == '__main__':
    unittest.main()
