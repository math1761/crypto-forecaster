from flask_restx import Api
from app.errors import handle_exception
from app.logger import setup_logger
from flask import Flask
from app.config import Config
from celery import Celery
from flask_caching import Cache

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    celery = Celery(
        app.import_name,
        broker=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0"
    )
    celery.conf.update(app.config)

    Cache(app, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': 'redis://redis:6379/0'})

    api = Api(app, version='1.0', title='Crypto Forecast API',
              description='API for predicting long-term cryptocurrency prices.')

    from app.routes.forecast import api as forecast_ns
    from app.routes.currencies import api as currencies_ns
    api.add_namespace(forecast_ns)
    api.add_namespace(currencies_ns)

    app.register_error_handler(Exception, handle_exception)

    setup_logger('app')

    return app
