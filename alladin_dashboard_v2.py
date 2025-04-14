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

# === Tickers to Track ===
TICKERS = ['DJT', 'WOLF', 'DOT', 'CL=F', 'NG=F', 'LMT', 'ETH-USD', 'BTC-USD', 'AAPL', 'TSLA']
ALWAYS_ON = ['ETH-USD', 'BTC-USD', 'CL=F', 'NG=F']  # crypto + commodities run 24/7

# === RSA Timezone Logic ===
def market_is_open(ticker):
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    rsa = pytz.timezone("Africa/Johannesburg")
    now = utc_now.astimezone(rsa)

    weekday = now.weekday()
    hour = now.hour
    minute = now.minute

    if weekday >= 5:
        return False

    # RSA market: 15:30–22:00 local time
    if ticker not in ALWAYS_ON:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)

    return True

# === Data Fetching ===
def fetch_data(ticker, interval='15m', period='2d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is None or df.empty or len(df) < 30:
            print(f"[{ticker}] No data available. Skipping.")
            return None
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])
        return df
    except Exception as e:
        print(f"[{ticker}] Error fetching data: {e}")
        return None

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

# === Signal Detection ===
def evaluate_signals(df, ticker):
    if df is None or len(df) < 30:
        return None

    last_index = df.index[-1]
    second_last_index = df.index[-2]

    try:
        close_now = df.at[last_index, 'Close']
        close_prev = df.at[second_last_index, 'Close']
        price_change = ((close_now - close_prev) / close_prev) * 100

        rsi = float(df.at[last_index, 'RSI'])
        macd = float(df.at[last_index, 'MACD'])
        signal = float(df.at[last_index, 'Signal'])
    except Exception as e:
        print(f"[{ticker}] Error extracting float values: {e}")
        return None

    signal_type = None
    reasons = []

    if price_change > 1.0 and rsi > 55 and macd > signal:
        signal_type = "STRONG BUY"
        reasons.append(f"RSI {rsi:.1f}")
        reasons.append("MACD crossover")
        reasons.append(f"+{price_change:.2f}%")
    elif price_change < -1.5 and rsi < 45 and macd < signal:
        signal_type = "STRONG SELL"
        reasons.append(f"RSI {rsi:.1f}")
        reasons.append("MACD downward")
        reasons.append(f"{price_change:.2f}%")
    elif macd > signal and price_change > 0.5:
        signal_type = "BUY"
        reasons.append("MACD rising")
        reasons.append(f"+{price_change:.2f}%")
    elif macd < signal and price_change < -0.5:
        signal_type = "SELL"
        reasons.append("MACD falling")
        reasons.append(f"{price_change:.2f}%")

    if signal_type:
        message = f"{ticker}: {signal_type} | {' | '.join(reasons)}"
        send_telegram_alert(message)
        return message

    return None

# === Alert System ===
def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Main Runtime ===
def main():
    now = dt.datetime.now(pytz.timezone('Africa/Johannesburg')).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Running Alladin Dashboard at RSA time: {now}")

    alerts = []

    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market is closed. Skipping.")
            continue

        df = fetch_data(ticker)
        result = evaluate_signals(df, ticker)
        if result:
            alerts.append(result)

    if alerts:
        print("\n--- Alerts ---")
        for alert in alerts:
            print(alert)
    else:
        print("No signals triggered.")

if __name__ == "__main__":
    main()
