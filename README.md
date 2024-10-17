# 📈 Crypto Forecast API

**Crypto Forecast API** is a Flask-based web API that predicts long-term cryptocurrency prices using [Facebook Prophet](https://facebook.github.io/prophet/). Whether you're looking to forecast Bitcoin, Ethereum, or any other cryptocurrency, this API has got you covered!

## 🌟 Features

- **Predict Price Trends**: Generates forecasts for cryptocurrencies over the next 90 days.
- **Multiple Cryptos**: Supports forecasting for a wide range of cryptocurrencies.
- **Fast & Scalable**: Built using Flask and can be deployed with Docker, making it lightweight and scalable.
- **Graphical Forecasts**: Generates and uploads prediction charts to AWS S3.
- **RESTful API**: Easy-to-use API endpoints that return results in JSON format.
  
## 🚀 Getting Started

### Prerequisites

To get the project up and running, you'll need to have the following:

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- AWS credentials (for uploading forecast images to S3)

### 🛠️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/crypto-forecast-api.git
    cd crypto-forecast-api
    ```

2. Set up your virtual environment and activate it:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure your environment variables. Create a `.env` file at the root of the project with the following:

    ```bash
    S3_BUCKET=your-s3-bucket-name
    S3_REGION=your-s3-region
    S3_ACCESS_KEY=your-aws-access-key
    S3_SECRET_KEY=your-aws-secret-key
    ```

### ⚙️ Usage

Once the setup is complete, you can run the Flask server locally:

```bash
flask run
```

### 🎯 API Endpoints

#### 1. Get Supported Cryptocurrencies

**Endpoint**: `/currencies`

**Description**: Returns a list of all supported cryptocurrencies.

**Requests**:

```bash
GET /currencies
```

**Response**:

```bash
{
  "currencies": [
    "bitcoin",
    "ethereum",
    "ripple",
    ...
  ]
}
```

#### 2. Predict Cryptocurrency Prices

**Endpoint**: `/forecast`

**Description**: Predicts the prices of the specified cryptocurrency for the next 90 days.

**Request**:

```bash
GET /forecast?currency=bitcoin&currency=ethereum
```

**Parameters**:

currency: (Optional) Specify one or more cryptocurrencies by their ID (e.g., bitcoin, ethereum). If no currency is provided, the API will use all supported cryptocurrencies.

**Response**:

```bash
{
  "forecasts": [
    {
      "currency": "bitcoin",
      "forecast": [
        {
          "ds": "2024-12-01",
          "yhat": 45000.12,
          "yhat_lower": 44000.56,
          "yhat_upper": 46000.91
        },
        {
          "ds": "2024-12-02",
          "yhat": 45500.34,
          "yhat_lower": 44500.78,
          "yhat_upper": 46500.67
        },
        ...
      ],
      "image_url": "https://your-bucket.s3.amazonaws.com/bitcoin_forecast_plot.png"
    },
    {
      "currency": "ethereum",
      "forecast": [
        {
          "ds": "2024-12-01",
          "yhat": 3000.12,
          "yhat_lower": 2900.56,
          "yhat_upper": 3100.91
        },
        ...
      ],
      "image_url": "https://your-bucket.s3.amazonaws.com/ethereum_forecast_plot.png"
    }
  ]
}
```

### 🐳 Docker

You can also run the app using Docker for a more portable solution:

1. Build the Docker image:

    ```bash
    docker build -t crypto-forecast-api .
    ```

2. Run the container:

    ```bash
    docker run -p 5000:5000 crypto-forecast-api
    ```

3. The app will be available at `http://localhost:5000`.

### 🧪 Running Tests

This project includes unit tests to ensure everything works as expected. To run the tests:

```bash
pytest
```