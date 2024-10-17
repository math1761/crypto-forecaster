import unittest
from app import create_app
from unittest.mock import patch
import json

class ForecastTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    @patch('app.services.currency_service.get_supported_currencies')
    @patch('app.services.prediction_service.fetch_historical_data')
    @patch('app.services.prediction_service.generate_forecast')
    @patch('app.services.s3_service.upload_image_to_s3')
    def test_forecast_endpoint(self, mock_upload, mock_generate, mock_fetch, mock_get_currencies):
        mock_get_currencies.return_value = ['bitcoin']
        mock_fetch.return_value = 'historical_data'
        mock_generate.return_value = [
            {"ds": "2024-12-01", "yhat": 45000.1234, "yhat_lower": 44000.5678, "yhat_upper": 46000.9101}
        ]
        mock_upload.return_value = 'https://s3.amazonaws.com/zappa-me5zu7zhj/bitcoin_forecast_plot.png'

        response = self.client.get('/forecast')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn('forecasts', data)
        self.assertEqual(len(data['forecasts']), 1)
        self.assertEqual(data['forecasts'][0]['currency'], 'bitcoin')
        self.assertIn('forecast', data['forecasts'][0])
        self.assertIn('image_url', data['forecasts'][0])

if __name__ == '__main__':
    unittest.main()
