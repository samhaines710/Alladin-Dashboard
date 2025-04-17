import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import pytz

# === Telegram Setup ===
try:
    import telegram
    TELEGRAM_ENABLED = True
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
except ImportError:
    print("Telegram module not found. Alerts disabled.")
    TELEGRAM_ENABLED = False
    bot = None

# === Ticker Categories ===
COMMODITIES = ['CL=F', 'NG=F', 'ETH-USD', 'BTC-USD']
STOCKS = ['DJT', 'WOLF', 'LMT', 'AAPL', 'TSLA', 'DOT']
ETFS = ['SPY', 'IVV']
TICKERS = COMMODITIES + STOCKS + ETFS
ALWAYS_ON = COMMODITIES + ETFS

# === News Sentiment (Mock Logic) ===
def get_news_sentiment(ticker):
    preset = {
        'DJT': 'Bearish',
        'WOLF': 'Neutral',
        'LMT': 'Neutral',
        'AAPL': 'Bullish',
        'TSLA': 'Neutral',
        'DOT': 'Neutral',
        'SPY': 'Neutral',
        'IVV': 'Neutral'
    }
    return preset.get(ticker, "Neutral")

# === Market Time Filter ===
def market_is_open(ticker):
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    rsa = pytz.timezone("Africa/Johannesburg")
    now = utc_now.astimezone(rsa)
    wd = now.weekday()
    hour = now.hour
    minute = now.minute
    if wd >= 5:
        return False
    if ticker in STOCKS or ticker in ETFS:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)
    return True

# === Indicators ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period=14):
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['range1'] = df['High'] - df['Low']
    df['range2'] = (df['High'] - df['prev_close']).abs()
    df['range3'] = (df['Low'] - df['prev_close']).abs()
    df['tr'] = df[['range1','range2','range3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']

# === Telegram Alert ===
def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Main ===
def main():
    local_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    print(f"Running Alladin at RSA time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")

    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market closed. Skipping.")
            continue

        try:
            df = yf.download(ticker, interval='5m', period='2d', progress=False)
            if df is None or df.empty or len(df) < 30:
                print(f"[{ticker}] No data or insufficient rows.")
                continue

            df['returns'] = df['Close'].pct_change()
            df['RSI'] = compute_rsi(df['Close'])
            macd_series, signal_series = compute_macd(df['Close'])
            df['MACD'] = macd_series
            df['Signal'] = signal_series
            df.dropna(inplace=True)

            # Apex logic (momentum peak + MACD histogram fade)
            if len(df) >= 3:
                macd_hist = df['MACD'] - df['Signal']
                if macd_hist.iloc[-2] > macd_hist.iloc[-1] and macd_hist.iloc[-3] < macd_hist.iloc[-2]:
                    send_telegram_alert(f"{ticker}: APEX WARNING | MACD histogram fading | Possible reversal")

            # RSI divergence (simplified)
            if df['RSI'].iloc[-1] > df['RSI'].iloc[-2] and df['Close'].iloc[-1] < df['Close'].iloc[-2]:
                send_telegram_alert(f"{ticker}: RSI DIVERGENCE | Price dropping while RSI rising | Watch for reversal")

            # Volatility Squeeze (Bollinger Bands logic)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['stddev'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + 2*df['stddev']
            df['Lower'] = df['MA20'] - 2*df['stddev']
            bb_width = df['Upper'] - df['Lower']
            if bb_width.iloc[-1] < bb_width.iloc[-5:].mean() * 0.75:
                send_telegram_alert(f"{ticker}: VOLATILITY SQUEEZE | Bollinger Band tightness | BIGGER MOVE COMING")

        except Exception as e:
            print(f"[{ticker}] Data error: {e}")

if __name__ == "__main__":
    main()
