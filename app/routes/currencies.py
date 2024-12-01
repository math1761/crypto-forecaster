from flask_restx import Namespace, Resource
from app.services.currency_service import get_supported_currencies
from app.logger import setup_logger

logger = setup_logger('currencies_route')

api = Namespace('currencies', description='Currency operations')

@api.route('/')
class CurrenciesList(Resource):
    def get(self):
        """List all supported cryptocurrencies"""
        currencies = get_supported_currencies()
        if not currencies:
            return {"error": "Aucune crypto-monnaie disponible"}, 500
        return {"currencies": currencies}, 200
