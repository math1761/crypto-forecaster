from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class ForecastData(BaseModel):
    ds: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float

class CurrencyForecast(BaseModel):
    currency: str
    forecast: List[ForecastData]
    image_url: Optional[str] = None
    error: Optional[str] = None
