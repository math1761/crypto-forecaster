from flask import Blueprint, jsonify
from app.services.currency_service import get_supported_currencies
from app.logger import setup_logger

logger = setup_logger('currencies_route')

currencies_bp = Blueprint('currencies_bp', __name__)

@currencies_bp.route('/currencies', methods=['GET'])
def list_currencies():
    currencies = get_supported_currencies()
    if not currencies:
        return jsonify({"error": "Aucune crypto-monnaie disponible"}), 500
    return jsonify({"currencies": currencies}), 200
