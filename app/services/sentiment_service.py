import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from app.logger import setup_logger
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

logger = setup_logger('sentiment_service')

def create_empty_sentiment_analysis(include_raw_data: bool = False) -> Dict:
    """Create an empty sentiment analysis structure"""
    sentiment = {
        'average_sentiment': 0.0,
        'sentiment_trend': 'neutral',
        'confidence': 0.0,
        'news_count': 0,
        'technical_bias': 0.0,
        'market_sentiment': 0.0,
        'detailed_metrics': {
            'news_sentiment': 0.0,
            'recent_sentiment': 0.0,
            'technical_indicators': {
                'bias': 0.0,
                'confidence': 0.0
            },
            'market_metrics': {
                'price_changes': {},
                'volume_metrics': {},
                'additional_metrics': {},
                'community_metrics': {
                    'twitter': {},
                    'reddit': {}
                },
                'developer_metrics': {}
            },
            'sentiment_distribution': {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            },
            'time_analysis': {
                'last_24h': {
                    'sentiment': 0.0,
                    'count': 0
                },
                'trend_strength': 0.0,
                'trend_direction': 'neutral'
            },
            'news_sources': {
                'count_by_source': {},
                'sentiment_by_source': {}
            }
        }
    }
    
    if include_raw_data:
        sentiment['raw_data'] = {
            'news_articles': [],
            'price_data': [],
            'technical_signals': {},
            'market_data': {},
            'community_data': {},
            'developer_data': {}
        }
    
    return sentiment

def fetch_news_data(currency: str) -> List[Dict]:
    """Fetch news data from multiple sources"""
    try:
        news_data = []
        
        # Reddit data
        try:
            reddit_url = f"https://www.reddit.com/r/CryptoCurrency/search.json"
            params = {
                'q': currency,
                'restrict_sr': 'on',
                'sort': 'new',
                'limit': 100
            }
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(reddit_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    news_data.append({
                        'source': 'Reddit',
                        'title': post_data.get('title', ''),
                        'description': post_data.get('selftext', ''),
                        'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'sentiment': analyze_sentiment(post_data.get('title', ''))
                    })
        except Exception as e:
            logger.warning(f"Error fetching Reddit data: {e}")

        # Messari data
        try:
            messari_url = f"https://data.messari.io/api/v1/news/{currency}"
            response = requests.get(messari_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('data', []):
                    news_data.append({
                        'source': 'Messari',
                        'title': article.get('title', ''),
                        'description': article.get('content', ''),
                        'published_at': article.get('published_at', ''),
                        'url': article.get('url', ''),
                        'sentiment': analyze_sentiment(article.get('title', ''))
                    })
        except Exception as e:
            logger.warning(f"Error fetching Messari data: {e}")

        return news_data

    except Exception as e:
        logger.error(f"Error fetching news data: {e}")
        return []

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text using a simple approach"""
    try:
        # Simple word-based sentiment analysis
        positive_words = {'bullish', 'buy', 'up', 'moon', 'gain', 'profit', 'success', 'surge', 'rise', 'high', 'growth'}
        negative_words = {'bearish', 'sell', 'down', 'crash', 'loss', 'fail', 'drop', 'low', 'fall', 'decline'}
        
        text = text.lower()
        words = set(text.split())
        
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
            
        total = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total
        
        return max(-1.0, min(1.0, sentiment))
        
    except Exception as e:
        logger.warning(f"Error analyzing sentiment: {e}")
        return 0.0

def analyze_news_sources(news_data: List[Dict]) -> Dict:
    """Analyze news sources statistics"""
    try:
        count_by_source = {}
        sentiment_by_source = defaultdict(list)
        
        for article in news_data:
            source = article.get('source', '')
            sentiment = article.get('sentiment', 0.0)
            
            count_by_source[source] = count_by_source.get(source, 0) + 1
            sentiment_by_source[source].append(sentiment)
        
        # Calculate average sentiment by source
        avg_sentiment = {
            source: float(np.mean(sentiments))
            for source, sentiments in sentiment_by_source.items()
        }
        
        return {
            'count_by_source': count_by_source,
            'sentiment_by_source': avg_sentiment
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing news sources: {e}")
        return {
            'count_by_source': {},
            'sentiment_by_source': {}
        }

def calculate_technical_bias(currency: str) -> Tuple[float, float]:
    """Calculate technical bias based on price data"""
    try:
        # Fetch historical price data
        df = fetch_historical_data(currency)
        if df.empty:
            return 0.0, 0.0

        # Calculate technical indicators using vectorized operations
        close_prices = df['close'].values
        
        # Calculate SMA
        sma_short = pd.Series(close_prices).rolling(window=20).mean().fillna(0).values
        sma_long = pd.Series(close_prices).rolling(window=50).mean().fillna(0).values
        ma_signal = np.where(sma_short > sma_long, 1, -1)
        
        # Calculate RSI
        delta = pd.Series(close_prices).diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
        
        # Calculate Momentum
        momentum = pd.Series(close_prices).pct_change(periods=10).fillna(0).values
        momentum_signal = np.where(momentum > 0, 1, -1)
        
        # Calculate Volatility
        volatility = pd.Series(close_prices).pct_change().rolling(window=20).std().fillna(0).values
        volatility_signal = np.where(volatility > volatility.mean(), -0.5, 0.5)
        
        # Combine signals with different weights
        weights = np.array([0.3, 0.3, 0.2, 0.2])  # MA, RSI, Momentum, Volatility
        signals = np.array([
            ma_signal[-1],
            rsi_signal[-1],
            momentum_signal[-1],
            volatility_signal[-1]
        ])
        
        technical_bias = float(np.dot(signals, weights))
        
        # Calculate confidence based on signal agreement
        signal_agreement = np.mean(np.abs(signals))
        confidence = float(min(1.0, signal_agreement))
        
        return technical_bias, confidence
        
    except Exception as e:
        logger.error(f"Error calculating technical bias for {currency}: {e}")
        return 0.0, 0.0

def get_sentiment_analysis(currency: str, include_raw_data: bool = False) -> Dict:
    """Get sentiment analysis for a cryptocurrency"""
    try:
        # Get news data and technical analysis in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            news_future = executor.submit(fetch_news_data, currency)
            tech_future = executor.submit(calculate_technical_bias, currency)
            
            news_data = news_future.result()
            technical_bias, technical_confidence = tech_future.result()
        
        if not news_data:
            return create_empty_sentiment_analysis(include_raw_data)
        
        # Calculate sentiment metrics
        sentiment_scores = [article.get('sentiment', 0) for article in news_data if article.get('sentiment') is not None]
        if not sentiment_scores:
            return create_empty_sentiment_analysis(include_raw_data)
        
        # Calculate average sentiment
        news_sentiment = float(np.mean(sentiment_scores))
        
        # Calculate sentiment distribution
        sentiment_distribution = {
            'positive': len([s for s in sentiment_scores if s > 0.2]),
            'neutral': len([s for s in sentiment_scores if -0.2 <= s <= 0.2]),
            'negative': len([s for s in sentiment_scores if s < -0.2])
        }
        
        # Calculate trend and confidence
        news_trend = 'positive' if news_sentiment > 0.2 else 'negative' if news_sentiment < -0.2 else 'neutral'
        news_confidence = float(min(1.0, len(sentiment_scores) / 100))  # Scale confidence based on number of articles
        
        # Calculate technical trend
        tech_trend = 'positive' if technical_bias > 0.2 else 'negative' if technical_bias < -0.2 else 'neutral'
        
        # Combine news and technical analysis with weights
        news_weight = 0.6  # 60% weight to news sentiment
        tech_weight = 0.4  # 40% weight to technical analysis
        
        combined_sentiment = (news_sentiment * news_weight) + (technical_bias * tech_weight)
        combined_confidence = (news_confidence * news_weight) + (technical_confidence * tech_weight)
        
        # Determine overall trend
        overall_trend = 'positive' if combined_sentiment > 0.2 else 'negative' if combined_sentiment < -0.2 else 'neutral'
        
        # Create technical indicators structure
        technical_indicators = {
            'bias': float(technical_bias),
            'confidence': float(technical_confidence)
        }
        
        # Create market metrics structure
        market_metrics = {
            'price_changes': {},
            'volume_metrics': {},
            'additional_metrics': {},
            'community_metrics': {
                'twitter': {},
                'reddit': {}
            },
            'developer_metrics': {}
        }
        
        # Create time analysis structure
        time_analysis = {
            'last_24h': {
                'sentiment': float(news_sentiment),
                'count': len(sentiment_scores)
            },
            'trend_strength': abs(float(combined_sentiment)),
            'trend_direction': overall_trend
        }
        
        # Create news sources analysis
        news_sources = analyze_news_sources(news_data)
        
        result = {
            'average_sentiment': float(combined_sentiment),
            'sentiment_trend': overall_trend,
            'confidence': float(combined_confidence),
            'news_count': len(news_data),
            'technical_bias': float(technical_bias),
            'market_sentiment': 0.0,
            'detailed_metrics': {
                'news_sentiment': float(news_sentiment),
                'recent_sentiment': float(news_sentiment),
                'technical_indicators': technical_indicators,
                'market_metrics': market_metrics,
                'sentiment_distribution': sentiment_distribution,
                'time_analysis': time_analysis,
                'news_sources': news_sources
            }
        }
        
        if include_raw_data:
            result['raw_data'] = {
                'news_articles': news_data,
                'price_data': [],
                'technical_signals': technical_indicators,
                'market_data': {},
                'community_data': {},
                'developer_data': {}
            }
        
        logger.info(f"Sentiment analysis for {currency}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis for {currency}: {e}")
        return create_empty_sentiment_analysis(include_raw_data)
