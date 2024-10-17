import matplotlib.pyplot as plt
from prophet import Prophet
import io
from app.logger import setup_logger
import pandas as pd

logger = setup_logger('plot_service')

def generate_plot(m: Prophet, forecast: pd.DataFrame, currency: str) -> io.BytesIO:
    fig = m.plot(forecast)
    plt.title(f"{currency.upper()} Forecast", fontsize=20)
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    plt.close(fig)
    img_buffer.seek(0)
    logger.info(f"Generated plot for {currency}.")
    return img_buffer
