from flask_restx import Namespace, Resource, fields
from app.services.currency_service import get_supported_currencies
from app.services.prediction_service import fetch_historical_data, preprocess_data, generate_forecast
from app.services.s3_service import upload_image_to_s3
from app.models.forecast import ForecastData, CurrencyForecast
from prophet import Prophet
from app.logger import setup_logger
from flask import request
from app.services.plot_service import generate_plot
from app.config import Config
from flask_caching import Cache
from flask import Blueprint

cache = Cache()

logger = setup_logger('forecast_route')

api = Namespace('forecast', description='Prédictions des prix des crypto-monnaies')

forecast_data = api.model('ForecastData', {
    'ds': fields.DateTime(required=True, description='Date de la prévision'),
    'yhat': fields.Float(required=True, description='Prix prédit'),
    'yhat_lower': fields.Float(required=True, description='Borne inférieure'),
    'yhat_upper': fields.Float(required=True, description='Borne supérieure'),
})

currency_forecast = api.model('CurrencyForecast', {
    'currency': fields.String(required=True, description='Nom de la crypto-monnaie'),
    'forecast': fields.List(fields.Nested(forecast_data)),
    'image_url': fields.String(description='URL de l\'image de prévision'),
    'error': fields.String(description='Message d\'erreur')
})

forecast_bp = Blueprint('forecast_bp', __name__)

@forecast_bp.route('/forecast', methods=['GET'])
@cache.cached(timeout=3600, query_string=True)
class Forecast(Resource):
    @api.doc('Get price forecasts for cryptocurrencies')
    @api.expect(api.parser().add_argument('currency', type=str, action='append', help='Crypto-monnaie à prévoir'))
    @api.marshal_with(api.model('ForecastResponse', {
        'forecasts': fields.List(fields.Nested(currency_forecast))
    }), code=200)
    def get(self):
        """Obtenir les prévisions de prix des crypto-monnaies"""
        currencies = request.args.getlist('currency')
        if not currencies:
            currencies = get_supported_currencies()
            if not currencies:
                api.abort(400, "Aucune crypto-monnaie disponible")

        forecasts = []

        for currency in currencies:
            logger.info(f"Processing forecast for {currency}")
            try:
                df = fetch_historical_data(currency)
                df = preprocess_data(df)
                forecast_df = generate_forecast(df)

                # Convert forecast to list of ForecastData
                forecast_data = [ForecastData(**row) for _, row in forecast_df.iterrows()]

                # Générer et uploader le graphique
                m = Prophet(
                    seasonality_mode='multiplicative',
                    daily_seasonality=True,
                    yearly_seasonality=True
                )
                m.fit(df)
                future = m.make_future_dataframe(periods=Config.PROPHET_PERIODS, include_history=False)
                forecast = m.predict(future)

                img_buffer = generate_plot(m, forecast, currency)
                filename = f"{currency}_forecast_plot.png"
                image_url = upload_image_to_s3(img_buffer, filename)

                forecasts.append(
                    CurrencyForecast(
                        currency=currency,
                        forecast=forecast_data,
                        image_url=image_url
                    )
                )
            except Exception as e:
                logger.error(f"Failed to process forecast for {currency}: {e}")
                forecasts.append(
                    CurrencyForecast(
                        currency=currency,
                        forecast=[],
                        error=str(e)
                    )
                )

        return {"forecasts": [forecast.dict() for forecast in forecasts]}, 200
