import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import requests
import os

# === Safe Telegram Setup ===
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

# === Tickers to Watch ===
TICKERS = ['DJT', 'WOLF', 'DOT', 'CL=F', 'NG=F', 'LMT', 'ETH-USD', 'BTC-USD', 'AAPL', 'TSLA']
ALWAYS_ON = ['ETH-USD', 'BTC-USD', 'CL=F', 'NG=F']

def market_is_open():
    now = dt.datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    # NYSE market hours: 9:30 AM to 4:00 PM EST = 15:30 to 22:00 SAST
    if weekday >= 5:
        return False
    return (hour == 15 and minute >= 30) or (16 <= hour < 22)

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

def evaluate_signals(df, ticker):
    if df is None or len(df) < 30:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price_change = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
    rsi = latest['RSI']
    macd = latest['MACD']
    signal = latest['Signal']

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

def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

def main():
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Running Alladin Dashboard at {now}")
    
    market_open = market_is_open()
    alerts = []

    for ticker in TICKERS:
        if not market_open and ticker not in ALWAYS_ON:
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
