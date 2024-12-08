import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from app.config import Config
from app.logger import setup_logger
from app.models.forecast import ForecastData
import requests
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator, PSARIndicator, TRIXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ForceIndexIndicator, EaseOfMovementIndicator, MFIIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = setup_logger('prediction_service')

__all__ = [
    'fetch_historical_data',
    'preprocess_data',
    'generate_forecast',
    'analyze_volume_patterns',
    'confirm_trend',
    'detect_market_condition',
    'calculate_weighted_signal',
    'perform_backtesting'
]

# Technical indicator weights
INDICATOR_WEIGHTS = {
    'trend': {
        'sma_20': 0.15,    # Short-term trend
        'sma_50': 0.15,    # Medium-term trend
        'sma_100': 0.10,   # Long-term trend
        'ema_20': 0.15,    # Short-term momentum
        'macd': 0.15,      # Trend momentum
        'macd_signal': 0.10,  # Trend confirmation
        'psar': 0.10,      # Trend reversal
        'trix': 0.10       # Triple smoothed trend
    },
    'momentum': {
        'rsi': 0.25,       # Overall momentum
        'stoch_k': 0.20,   # Fast momentum
        'stoch_d': 0.15,   # Slow momentum
        'williams_r': 0.20,  # Oversold/overbought
        'roc': 0.20        # Rate of change
    },
    'volatility': {
        'bb_width': 0.35,  # Volatility measure
        'bb_position': 0.35,  # Price position
        'atr': 0.30        # Average true range
    },
    'volume': {
        'obv': 0.25,       # On-balance volume
        'volume_sma': 0.25,  # Volume trend
        'volume_ema': 0.25,  # Volume momentum
        'volume_roc': 0.25   # Volume change rate
    }
}

def fetch_historical_data(currency: str) -> pd.DataFrame:
    """Fetch historical price data for a cryptocurrency"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{currency}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '365',  # Get one year of data
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'prices' not in data:
            raise ValueError("No price data available")
            
        # Convert to DataFrame
        df = pd.DataFrame(data['prices'], columns=['ds', 'y'])
        
        # Convert millisecond timestamps to datetime
        df['ds'] = pd.to_datetime(df['ds'], unit='ms')
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {currency}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data with enhanced features"""
    try:
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            if df['ds'].dtype == np.int64 or df['ds'].dtype == np.float64:
                df['ds'] = pd.to_datetime(df['ds'], unit='ms')
            else:
                df['ds'] = pd.to_datetime(df['ds'])
        
        # Convert price to numeric
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Add technical indicators
        if len(df) >= 20:  # Ensure enough data for indicators
            # Trend indicators
            df['sma_20'] = SMAIndicator(close=df['y'], window=20, fillna=True).sma_indicator()
            df['ema_20'] = EMAIndicator(close=df['y'], window=20, fillna=True).ema_indicator()
            
            if len(df) >= 50:
                df['sma_50'] = SMAIndicator(close=df['y'], window=50, fillna=True).sma_indicator()
            
            if len(df) >= 100:
                df['sma_100'] = SMAIndicator(close=df['y'], window=100, fillna=True).sma_indicator()
            
            # MACD
            macd = MACD(close=df['y'], fillna=True)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # RSI
            df['rsi'] = RSIIndicator(close=df['y'], fillna=True).rsi()
            
            # Stochastic
            stoch = StochasticOscillator(high=df['y'], low=df['y'], close=df['y'], fillna=True)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['y'], fillna=True)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['y']
            
            # ATR
            df['atr'] = AverageTrueRange(high=df['y'], low=df['y'], close=df['y'], fillna=True).average_true_range()
            
            # Price momentum
            df['price_momentum'] = df['y'].pct_change().fillna(0)
            
            # Volatility
            df['volatility'] = df['price_momentum'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # Add cyclical features
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['year'] = df['ds'].dt.year
        df['day_of_year'] = df['ds'].dt.dayofyear
        df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(float)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Normalize indicators (excluding 'ds' and 'y')
        cols_to_normalize = [col for col in numeric_cols if col not in ['ds', 'y']]
        if cols_to_normalize:
            scaler = StandardScaler()
            df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        
        # Final check for any remaining NaN values
        df = df.fillna(0)
        
        logger.info("Successfully preprocessed data with enhanced features")
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def detect_market_condition(df: pd.DataFrame) -> str:
    """Detect market condition based on multiple indicators"""
    try:
        # Get recent data for analysis
        recent_data = df.tail(20)
        
        # Calculate trend indicators
        price = df['y'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else price
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0
        
        # Determine market condition
        if volatility > 1.5:  # High volatility
            return 'volatile'
        elif price > sma_20 and rsi > 50:
            return 'uptrend'
        elif price < sma_20 and rsi < 50:
            return 'downtrend'
        else:
            return 'ranging'
            
    except Exception as e:
        logger.warning(f"Error detecting market condition: {str(e)}, defaulting to 'ranging'")
        return 'ranging'

def analyze_volume_patterns(df: pd.DataFrame) -> Dict:
    """Analyze volume patterns and trends"""
    try:
        if 'volume' not in df.columns:
            return {'volume_trend': 0, 'patterns': []}
            
        recent_volume = df['volume'].tail(20).mean()
        historical_volume = df['volume'].mean()
        volume_trend = (recent_volume / historical_volume) - 1
        
        patterns = []
        if volume_trend > 0.5:
            patterns.append('volume_breakout')
        elif volume_trend < -0.5:
            patterns.append('volume_decline')
            
        return {
            'volume_trend': volume_trend,
            'patterns': patterns
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing volume patterns: {str(e)}")
        return {'volume_trend': 0, 'patterns': []}

def confirm_trend(df: pd.DataFrame, market_condition: str) -> Dict:
    """Confirm trend with multiple indicators"""
    try:
        signals = []
        
        # Price trend
        price_trend = np.sign(df['y'].diff().iloc[-1])
        signals.append(price_trend)
        
        # Moving average trend
        if 'sma_20' in df.columns:
            ma_trend = np.sign(df['y'].iloc[-1] - df['sma_20'].iloc[-1])
            signals.append(ma_trend)
            
        # RSI trend
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            rsi_signal = 1 if rsi > 60 else (-1 if rsi < 40 else 0)
            signals.append(rsi_signal)
            
        # MACD trend
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_trend = np.sign(df['macd'].iloc[-1] - df['macd_signal'].iloc[-1])
            signals.append(macd_trend)
            
        # Calculate confidence
        if signals:
            trend_score = np.mean(signals)
            confidence = abs(trend_score)
        else:
            trend_score = 0
            confidence = 0
            
        return {
            'score': float(trend_score),
            'confidence': float(confidence)
        }
        
    except Exception as e:
        logger.warning(f"Error confirming trend: {str(e)}")
        return {'score': 0.0, 'confidence': 0.0}

def calculate_weighted_signal(df: pd.DataFrame) -> float:
    """Calculate weighted signal from multiple indicators"""
    try:
        signals = []
        weights = []
        
        # Process trend indicators
        for indicator, weight in INDICATOR_WEIGHTS['trend'].items():
            if indicator in df.columns:
                try:
                    if indicator in ['sma_20', 'sma_50', 'sma_100', 'ema_20']:
                        signal = np.sign(df['y'].iloc[-1] - df[indicator].iloc[-1])
                    elif indicator in ['macd', 'macd_signal']:
                        signal = np.sign(df['macd'].iloc[-1] - df['macd_signal'].iloc[-1])
                    elif indicator == 'psar':
                        signal = np.sign(df['y'].iloc[-1] - df['psar'].iloc[-1])
                    elif indicator == 'trix':
                        signal = np.sign(df['trix'].iloc[-1])
                    
                    signals.append(signal)
                    weights.append(weight)
                except:
                    continue
        
        # Process momentum indicators
        for indicator, weight in INDICATOR_WEIGHTS['momentum'].items():
            if indicator in df.columns:
                try:
                    if indicator == 'rsi':
                        signal = 1 if df['rsi'].iloc[-1] > 70 else (-1 if df['rsi'].iloc[-1] < 30 else 0)
                    elif indicator in ['stoch_k', 'stoch_d']:
                        signal = 1 if df[indicator].iloc[-1] > 80 else (-1 if df[indicator].iloc[-1] < 20 else 0)
                    elif indicator == 'williams_r':
                        signal = 1 if df[indicator].iloc[-1] > -20 else (-1 if df[indicator].iloc[-1] < -80 else 0)
                    elif indicator == 'roc':
                        signal = np.sign(df[indicator].iloc[-1])
                    
                    signals.append(signal)
                    weights.append(weight)
                except:
                    continue
        
        # Calculate weighted average if we have signals
        if signals:
            weighted_signal = np.average(signals, weights=weights)
            return np.clip(weighted_signal, -1, 1)
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Error calculating weighted signal: {str(e)}, defaulting to 0")
        return 0.0

def perform_backtesting(df: pd.DataFrame, forecast_periods: int = 30) -> Dict:
    """Perform backtesting on historical data"""
    try:
        results = []
        test_size = forecast_periods
        
        # Test on multiple periods
        for i in range(4):
            try:
                test_start = -test_size * (i + 1)
                test_end = test_start + test_size if i > 0 else None
                
                train = df.iloc[:test_start].copy()
                test = df.iloc[test_start:test_end].copy()
                
                if len(train) < 50 or len(test) < test_size:
                    continue
                
                # Train Prophet model
                m = Prophet(
                    seasonality_mode='multiplicative',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
                
                train_prophet = train[['ds', 'y']].copy()
                test_prophet = test[['ds', 'y']].copy()
                
                m.fit(train_prophet)
                future = m.make_future_dataframe(periods=test_size)
                forecast = m.predict(future)
                
                # Calculate metrics
                y_true = test_prophet['y'].values
                y_pred = forecast.iloc[-test_size:]['yhat'].values
                
                mape = mean_absolute_percentage_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                results.append({
                    'period': i + 1,
                    'mape': mape,
                    'rmse': rmse,
                    'r2': r2
                })
                
            except Exception as e:
                logger.warning(f"Failed backtesting for period {i + 1}: {str(e)}")
                continue
        
        if not results:
            return {'mape': 1.0, 'rmse': float('inf'), 'r2': 0.0, 'test_periods': 0}
        
        return {
            'mape': np.mean([r['mape'] for r in results]),
            'rmse': np.mean([r['rmse'] for r in results]),
            'r2': np.mean([r['r2'] for r in results]),
            'test_periods': len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return {'mape': 1.0, 'rmse': float('inf'), 'r2': 0.0, 'test_periods': 0}

def generate_forecast(df: pd.DataFrame, currency: str) -> Tuple[pd.DataFrame, Dict]:
    """Generate forecast using Prophet"""
    try:
        # Ensure datetime format for Prophet
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Configure and fit Prophet model
        m = Prophet(
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        
        m.fit(df)
        
        # Generate future dates
        future = m.make_future_dataframe(periods=Config.PROPHET_PERIODS, include_history=False)
        forecast = m.predict(future)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(df, forecast, currency)
        
        return forecast, metrics
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise

def calculate_forecast_metrics(df: pd.DataFrame, forecast: pd.DataFrame, currency: str) -> Dict:
    """Calculate comprehensive forecast metrics"""
    try:
        # Calculate backtest metrics
        backtest_metrics = perform_backtesting(df)
        
        # Calculate current metrics
        current_price = df['y'].iloc[-1]
        forecast_end_price = forecast['yhat'].iloc[-1]
        price_change_pct = ((forecast_end_price - current_price) / current_price) * 100
        
        # Detect market condition
        market_condition = detect_market_condition(df)
        
        # Calculate confidence score based on backtest metrics
        confidence_score = 1 - backtest_metrics.get('mape', 1.0)
        confidence_score = max(0.0, min(1.0, confidence_score))  # Clip between 0 and 1
        
        metrics = {
            'backtest_metrics': backtest_metrics,
            'forecast_metrics': {
                'current_price': float(current_price),
                'forecast_end_price': float(forecast_end_price),
                'price_change_pct': float(price_change_pct),
                'forecast_period_days': Config.PROPHET_PERIODS,
                'trend': market_condition,
                'confidence_score': float(confidence_score)
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating forecast metrics: {e}")
        return {
            'backtest_metrics': {'mape': 1.0, 'rmse': float('inf'), 'r2': 0.0, 'test_periods': 0},
            'forecast_metrics': {
                'current_price': None,
                'forecast_end_price': None,
                'price_change_pct': None,
                'forecast_period_days': Config.PROPHET_PERIODS,
                'trend': 'unknown',
                'confidence_score': 0.0
            }
        }
