import pandas as pd
from prophet import Prophet
from datetime import datetime
from typing import List, Dict
from app.config import Config
from app.logger import setup_logger
from app.models.forecast import ForecastData
import requests

logger = setup_logger('prediction_service')

def fetch_historical_data(currency: str) -> pd.DataFrame:
    url = Config.COINCAP_HISTORY_URL.format(currency=currency)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        logger.info(f"Fetched historical data for {currency}.")
        return df
    except requests.RequestException as e:
        logger.error(f"Error fetching historical data for {currency}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"time": "ds", "priceUsd": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'].apply(lambda timestamp: datetime.fromtimestamp(int(timestamp) / 1000)))
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])
    logger.info("Preprocessed data for Prophet.")
    return df

def generate_forecast(df: pd.DataFrame) -> pd.DataFrame:
    m = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        yearly_seasonality=True
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=Config.PROPHET_PERIODS, include_history=False)
    forecast = m.predict(future)
    logger.info("Generated forecast using Prophet.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
