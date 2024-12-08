# Crypto Forecaster

A sophisticated cryptocurrency price prediction and analysis service that combines machine learning, technical analysis, and sentiment analysis to provide comprehensive market insights.

## Features

### 1. Price Forecasting
- Prophet-based time series forecasting
- 90-day price predictions with confidence intervals
- Backtesting with multiple validation periods
- Dynamic model adjustments based on market conditions
- Technical indicator integration for improved accuracy

### 2. Advanced Sentiment Analysis
- Multi-source news aggregation:
  - Reddit (multiple cryptocurrency subreddits)
  - CryptoPanic news aggregator
  - CoinGecko updates and market data
  - LunarCrush social metrics
  - Messari crypto research
- Sophisticated sentiment scoring:
  - Time-weighted sentiment analysis
  - Crypto-specific keyword recognition
  - Trend detection and confidence scoring
  - Social media engagement metrics

### 3. Technical Analysis
- Multiple timeframe analysis
- Key technical indicators:
  - Moving Averages (SMA, EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Volume analysis
- Trend strength measurement
- Market condition detection

### 4. Market Metrics
- Price change analysis across multiple timeframes
- Volume/Market cap analysis
- Community engagement metrics
- Social media trend analysis
- Market sentiment indicators

### 5. Real-time Updates
- Automatic data refresh
- Redis caching for performance
- Configurable update intervals
- Historical data analysis

## Technology Stack

- **Backend**: Python/Flask
- **Machine Learning**: Prophet, scikit-learn
- **Data Analysis**: pandas, numpy, ta
- **Caching**: Redis
- **Containerization**: Docker
- **API Documentation**: Swagger/OpenAPI
- **Cloud Storage**: AWS S3 (optional)

## Setup and Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Redis

### Local Development Setup

1. Clone the repository:
\`\`\`bash
git clone [repository-url]
cd crypto-forecaster
\`\`\`

2. Create and configure environment variables:
\`\`\`bash
cp .env.example .env
# Edit .env with your configurations
\`\`\`

3. Start the development environment:
\`\`\`bash
docker-compose -f docker-compose.dev.yml up --build
\`\`\`

4. Access the services:
- API: http://localhost:5000
- Swagger Documentation: http://localhost:5000/swagger
- Redis Commander: http://localhost:8081

### Environment Variables

\`\`\`env
FLASK_ENV=development
FLASK_DEBUG=True
REDIS_HOST=redis
REDIS_PORT=6379
S3_BUCKET=your-bucket-name
S3_REGION=your-region
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
\`\`\`

## API Endpoints

### GET /forecast/
Get cryptocurrency price forecasts with sentiment analysis

Parameters:
- \`currency\`: Cryptocurrency to analyze (e.g., bitcoin, ethereum, solana)

Response includes:
- Price forecasts for next 90 days
- Confidence intervals
- Sentiment analysis
- Technical indicators
- Market metrics
- Backtesting results

Example response:
\`\`\`json
{
  "forecasts": [{
    "currency": "bitcoin",
    "forecast": [...],
    "backtest_metrics": {
      "mape": 0.15,
      "rmse": 1000.0,
      "r2": 0.85,
      "test_periods": 4
    },
    "sentiment_analysis": {
      "average_sentiment": 0.65,
      "sentiment_trend": "improving",
      "confidence": 0.8,
      "news_count": 50,
      "technical_bias": 0.3,
      "market_sentiment": 0.4,
      "detailed_metrics": {
        "news_sentiment": 0.65,
        "recent_sentiment": 0.7,
        "technical_indicators": {
          "bias": 0.3,
          "confidence": 0.8
        },
        "market_metrics": {
          "sentiment": 0.4,
          "confidence": 0.7
        },
        "sentiment_distribution": {
          "positive": 30,
          "neutral": 15,
          "negative": 5
        }
      }
    },
    "forecast_metrics": {
      "current_price": 35000.0,
      "forecast_end_price": 42000.0,
      "price_change_pct": 20.0,
      "forecast_period_days": 90,
      "trend": "uptrend",
      "confidence_score": 0.85
    }
  }]
}
\`\`\`

## Docker Support

### Development Environment
- Hot-reloading enabled
- Debug mode
- Redis Commander included
- Volume mounting for live code updates

### Production Environment
- Optimized for performance
- Redis persistence
- Health checks
- Automatic restarts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details