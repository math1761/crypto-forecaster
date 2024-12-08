from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime

class ForecastData(BaseModel):
    ds: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float

class BacktestMetrics(BaseModel):
    mape: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    test_periods: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

class RawData(BaseModel):
    news_articles: List[Dict] = []
    price_data: List[Dict] = []
    technical_signals: Dict = {}
    market_data: Dict = {}
    community_data: Dict = {}
    developer_data: Dict = {}

class DetailedMetrics(BaseModel):
    news_sentiment: float = 0.0
    recent_sentiment: float = 0.0
    technical_indicators: Dict = {}
    market_metrics: Dict = {}
    sentiment_distribution: Dict = {}
    time_analysis: Dict = {}
    news_sources: Dict = {}

class SentimentAnalysis(BaseModel):
    average_sentiment: float = 0.0
    sentiment_trend: str = 'neutral'
    confidence: float = 0.0
    news_count: int = 0
    technical_bias: float = 0.0
    market_sentiment: float = 0.0
    raw_data: RawData = RawData()
    detailed_metrics: DetailedMetrics = DetailedMetrics()

    class Config:
        arbitrary_types_allowed = True

class ForecastMetrics(BaseModel):
    current_price: Optional[float] = None
    forecast_end_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    forecast_period_days: Optional[int] = None
    trend: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class CurrencyForecast(BaseModel):
    currency: str
    forecast: List[ForecastData] = []
    image_url: Optional[str] = None
    error: Optional[str] = None
    backtest_metrics: Optional[BacktestMetrics] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    forecast_metrics: Optional[ForecastMetrics] = None

    class Config:
        arbitrary_types_allowed = True
