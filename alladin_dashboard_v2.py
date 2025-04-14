import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import requests
import os
import telegram

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# === Track These Tickers ===
TICKERS = ['DJT', 'WOLF', 'DOT', 'CL=F', 'NG=F', 'LMT', 'ETH-USD', 'BTC-USD', 'AAPL', 'TSLA']

def fetch_data(ticker, interval='5m', period='1d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
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
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")
    else:
        print("Telegram not configured. Skipping alert.")

def main():
    print(f"Running Alladin Dashboard at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    alerts = []

    for ticker in TICKERS:
        df = fetch_data(ticker)
        result = evaluate_signals(df, ticker)
        if result:
            alerts.append(result)

    print("\n--- Alerts ---")
    for alert in alerts:
        print(alert)

if __name__ == "__main__":
    main()
