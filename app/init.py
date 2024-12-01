from flask_restx import Api
from app.errors import handle_exception
from app.logger import setup_logger
from flask import Flask, request, g
from app.config import Config
from celery import Celery
from flask_caching import Cache
import time

# Initialize cache
cache = Cache()
logger = setup_logger('app')

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize Celery
    celery = Celery(
        app.import_name,
        broker=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0"
    )
    celery.conf.update(app.config)

    # Initialize cache with app
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = f'redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0'
    cache.init_app(app)

    # Request timing middleware
    @app.before_request
    def start_timer():
        g.start = time.time()

    @app.after_request
    def log_request_time(response):
        if hasattr(g, 'start'):
            total_time = (time.time() - g.start) * 1000  # Convert to milliseconds
            logger.info(f"{request.method} {request.path} {response.status_code} - {total_time:.2f}ms")
        return response

    # Initialize API
    api = Api(app, version='1.0', title='Crypto Forecast API',
              description='API for predicting long-term cryptocurrency prices.')

    from app.routes.forecast import api as forecast_ns
    from app.routes.currencies import api as currencies_ns
    api.add_namespace(forecast_ns)
    api.add_namespace(currencies_ns)

    app.register_error_handler(Exception, handle_exception)

    return app
