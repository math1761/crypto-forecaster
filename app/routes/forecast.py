from flask_restx import Namespace, Resource, fields, inputs
from app.services.currency_service import get_supported_currencies
from app.services.prediction_service import fetch_historical_data, preprocess_data, generate_forecast
from app.services.s3_service import upload_image_to_s3, is_s3_configured
from app.models.forecast import ForecastData, CurrencyForecast, BacktestMetrics, SentimentAnalysis, ForecastMetrics
from app.services.sentiment_service import get_sentiment_analysis, create_empty_sentiment_analysis
from prophet import Prophet
from app.logger import setup_logger
from flask import request, current_app
from app.services.plot_service import generate_plot
from app.config import Config
from app.init import cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import os
from functools import partial
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import numpy as np
import time
import pandas as pd
from typing import Optional

logger = setup_logger('forecast_route')

# Initialize pools
NUM_CORES = multiprocessing.cpu_count()
THREAD_POOL = ThreadPoolExecutor(max_workers=NUM_CORES * 4)  # More threads for I/O
PROCESS_POOL = ProcessPoolExecutor(max_workers=NUM_CORES)    # One process per core

# Configure session with retries and connection pooling
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100))

# Cache for intermediate results
DATA_CACHE = {}

def get_cached_or_fetch(key, fetch_func, cache_time=300):  # 5 minutes cache
    """Get data from cache or fetch it"""
    if key in DATA_CACHE:
        result, timestamp = DATA_CACHE[key]
        if time.time() - timestamp < cache_time:
            return result
    result = fetch_func()
    DATA_CACHE[key] = (result, time.time())
    return result

def parallel_fetch_data(urls):
    """Fetch multiple URLs in parallel"""
    def fetch_single_url(url_info):
        try:
            response = session.get(
                url_info['url'], 
                params=url_info.get('params'),
                headers=url_info.get('headers'),
                timeout=10
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {url_info['url']}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = [executor.submit(fetch_single_url, url) for url in urls]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def process_data_chunk(chunk):
    """Process a chunk of data in parallel"""
    try:
        return preprocess_data(chunk)
    except Exception as e:
        logger.error(f"Error processing data chunk: {e}")
        return None

def split_dataframe(df, num_chunks):
    """Split dataframe into chunks for parallel processing"""
    chunk_size = len(df) // num_chunks
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def process_single_currency(currency: str, include_raw_data: bool = False) -> CurrencyForecast:
    """Process forecast for a single currency with optimized parallel processing"""
    try:
        # Fetch historical data
        df = fetch_historical_data(currency)
        if df is None or df.empty:
            return create_empty_forecast(currency, include_raw_data)

        # Process data and generate forecast in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Start sentiment analysis early as it's network-bound
            sentiment_future = executor.submit(get_sentiment_analysis, currency, include_raw_data)
            
            # Process data and generate forecast (CPU-bound)
            df = preprocess_data(df)
            forecast_df, metrics = generate_forecast(df, currency)
            
            # Get sentiment results
            try:
                sentiment_result = sentiment_future.result(timeout=10)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.error(f"Sentiment analysis failed: {e}")
                sentiment_result = create_empty_sentiment_analysis(include_raw_data)
            
            # Update metrics with sentiment analysis
            metrics['sentiment_analysis'] = sentiment_result
            
            # Convert forecast to list of ForecastData
            forecast_data = [
                ForecastData(
                    ds=row['ds'],
                    yhat=row['yhat'],
                    yhat_lower=row['yhat_lower'],
                    yhat_upper=row['yhat_upper']
                ) for _, row in forecast_df.iterrows()
            ]
            
            # Generate plot if needed
            image_url = None
            if is_s3_configured():
                try:
                    plot_future = executor.submit(generate_and_upload_plot, df, currency)
                    image_url = plot_future.result(timeout=10)
                except (concurrent.futures.TimeoutError, Exception) as e:
                    logger.error(f"Plot generation failed: {e}")

            return CurrencyForecast(
                currency=currency,
                forecast=forecast_data,
                image_url=image_url,
                backtest_metrics=BacktestMetrics(**metrics['backtest_metrics']),
                sentiment_analysis=SentimentAnalysis(**metrics['sentiment_analysis']),
                forecast_metrics=ForecastMetrics(**metrics['forecast_metrics'])
            )

    except Exception as e:
        logger.error(f"Failed to process forecast for {currency}: {e}")
        return create_empty_forecast(currency, include_raw_data)

def create_empty_forecast(currency: str, include_raw_data: bool = False) -> CurrencyForecast:
    """Create an empty forecast structure"""
    empty_sentiment = {
        'average_sentiment': None,
        'sentiment_trend': 'neutral',
        'confidence': None,
        'news_count': None,
        'raw_data': {
            'news_articles': [],
            'price_data': None,
            'technical_signals': None,
            'market_data': None,
            'community_data': None,
            'developer_data': None
        } if include_raw_data else None
    }
    
    return CurrencyForecast(
        currency=currency,
        forecast=[],
        error="Failed to process forecast",
        backtest_metrics=BacktestMetrics(),  # All fields will be None by default
        sentiment_analysis=SentimentAnalysis(**empty_sentiment),
        forecast_metrics=ForecastMetrics()  # All fields will be None by default
    )

def generate_and_upload_plot(df: pd.DataFrame, currency: str) -> Optional[str]:
    """Generate and upload plot to S3"""
    try:
        m = Prophet(
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=Config.PROPHET_PERIODS, include_history=False)
        forecast = m.predict(future)
        
        img_buffer = generate_plot(m, forecast, currency)
        filename = f"{currency}_forecast_plot.png"
        return upload_image_to_s3(img_buffer, filename)
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        return None

api = Namespace('forecast', description='Cryptocurrency price predictions')

# Define Swagger models
forecast_data = api.model('ForecastData', {
    'ds': fields.DateTime(required=True, description='Forecast date'),
    'yhat': fields.Float(required=True, description='Predicted price'),
    'yhat_lower': fields.Float(required=True, description='Lower bound of prediction interval'),
    'yhat_upper': fields.Float(required=True, description='Upper bound of prediction interval')
})

backtest_metrics = api.model('BacktestMetrics', {
    'mape': fields.Float(description='Mean Absolute Percentage Error'),
    'rmse': fields.Float(description='Root Mean Square Error'),
    'r2': fields.Float(description='R-squared score'),
    'test_periods': fields.Integer(description='Number of test periods')
})

raw_data_model = api.model('RawData', {
    'news_articles': fields.List(fields.Raw, description='Raw news articles data'),
    'price_data': fields.List(fields.Raw, description='Historical price data'),
    'technical_signals': fields.Raw(description='Technical analysis signals and indicators'),
    'market_data': fields.Raw(description='Market-related data'),
    'community_data': fields.Raw(description='Social media and community metrics'),
    'developer_data': fields.Raw(description='Developer activity metrics')
})

sentiment_analysis = api.model('SentimentAnalysis', {
    'average_sentiment': fields.Float(description='Average sentiment score (-1 to 1)'),
    'sentiment_trend': fields.String(description='Overall sentiment trend'),
    'confidence': fields.Float(description='Confidence in sentiment analysis'),
    'news_count': fields.Integer(description='Number of news articles analyzed'),
    'raw_data': fields.Nested(raw_data_model, description='Raw data used for analysis (if requested)', required=False)
})

forecast_metrics = api.model('ForecastMetrics', {
    'current_price': fields.Float(description='Current price'),
    'forecast_end_price': fields.Float(description='Forecasted end price'),
    'price_change_pct': fields.Float(description='Predicted price change percentage'),
    'forecast_period_days': fields.Integer(description='Number of days forecasted'),
    'trend': fields.String(description='Overall price trend'),
    'confidence_score': fields.Float(description='Model confidence score')
})

currency_forecast = api.model('CurrencyForecast', {
    'currency': fields.String(required=True, description='Cryptocurrency name'),
    'forecast': fields.List(fields.Nested(forecast_data), description='List of price predictions'),
    'image_url': fields.String(description='URL of the forecast plot image (only available if S3 is configured)'),
    'error': fields.String(description='Error message if prediction failed'),
    'backtest_metrics': fields.Nested(backtest_metrics, description='Backtesting performance metrics'),
    'sentiment_analysis': fields.Nested(sentiment_analysis, description='News sentiment analysis'),
    'forecast_metrics': fields.Nested(forecast_metrics, description='Additional forecast metrics')
})

forecast_response = api.model('ForecastResponse', {
    'forecasts': fields.List(fields.Nested(currency_forecast), description='List of forecasts for requested cryptocurrencies')
})

forecast_parser = api.parser()
forecast_parser.add_argument('currency', type=str, action='append', help='Cryptocurrency to forecast (e.g., bitcoin, ethereum). Can be specified multiple times. If not specified, all supported currencies will be used.')
forecast_parser.add_argument('include_raw_data', type=inputs.boolean, default=False, help='Include raw data in sentiment analysis (default: false)')

@api.route('/')
class Forecast(Resource):
    @api.doc('get_forecasts',
             description='''Get price forecasts for cryptocurrencies over the next 90 days.''')
    @api.expect(forecast_parser)
    @api.marshal_with(forecast_response, code=200)
    @cache.cached(timeout=1800, query_string=True)  # 30 minutes cache
    def get(self):
        """Get cryptocurrency price forecasts"""
        currencies = request.args.getlist('currency')
        include_raw_data = request.args.get('include_raw_data', 'false').lower() == 'true'
        
        if not currencies:
            currencies = get_supported_currencies()
            if not currencies:
                api.abort(400, "No cryptocurrencies available")

        # Process currencies in parallel using ThreadPoolExecutor for better I/O handling
        with ThreadPoolExecutor(max_workers=min(32, len(currencies) * 2)) as executor:
            futures = [
                executor.submit(process_single_currency, currency, include_raw_data)
                for currency in currencies
            ]
            
            forecasts = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    forecast = future.result(timeout=30)
                    forecasts.append(forecast)
                except concurrent.futures.TimeoutError:
                    logger.error("Forecast processing timed out")
                    continue
                except Exception as e:
                    logger.error(f"Error processing forecast: {e}")
                    continue

        return {"forecasts": [forecast.dict() for forecast in forecasts]}, 200
