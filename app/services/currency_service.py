import requests
from typing import List
from app.config import Config
from app.logger import setup_logger

logger = setup_logger('currency_service')

def get_supported_currencies() -> List[str]:
    try:
        response = requests.get(Config.COINCAP_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        currencies = [coin['id'] for coin in data.get('data', []) if 'id' in coin]
        logger.info(f"Retrieved {len(currencies)} currencies.")
        return currencies
    except requests.RequestException as e:
        logger.error(f"Error fetching supported currencies: {e}")
        return []
