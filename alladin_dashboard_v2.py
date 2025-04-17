# Alladin Ultimate v2 - Apex Enhanced
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import os

# === Telegram Setup ===
try:
    import telegram
    TELEGRAM_ENABLED = True
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
except ImportError:
    TELEGRAM_ENABLED = False
    bot = None

COMMODITIES = ['CL=F', 'NG=F']
STOCKS = ['DJT', 'WOLF', 'LMT', 'AAPL', 'TSLA', 'DOT']
ETFS = ['SPY', 'IVV']
TICKERS = COMMODITIES + STOCKS + ETFS
signal_log = {}

# === Helpers ===
def market_is_open(ticker):
    now = dt.datetime.now(pytz.timezone("Africa/Johannesburg"))
    if now.weekday() >= 5:
        return False
    if ticker in STOCKS + ETFS:
        return 15 <= now.hour < 22
    return True

def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Indicator Logic ===
def compute_indicators(df):
    df['returns'] = df['Close'].pct_change()
    df['RSI'] = df['Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / \
                df['Close'].diff().abs().rolling(14).mean() * 100
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    df['UpperBB'] = df['Close'].rolling(window=20).mean() + 2*df['Close'].rolling(window=20).std()
    df['LowerBB'] = df['Close'].rolling(window=20).mean() - 2*df['Close'].rolling(window=20).std()
    df['BBWidth'] = df['UpperBB'] - df['LowerBB']
    df['BBWidthNorm'] = df['BBWidth'].rolling(20).mean()
    return df.dropna()

def fetch_data(ticker, interval='5m', period='2d'):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty or len(df) < 20:
        return None
    return compute_indicators(df)

def evaluate_signal(df, ticker):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev3 = df.iloc[-3]
    rsi = latest['RSI']
    macd, signal = latest['MACD'], latest['Signal']
    price_now = latest['Close']
    price_prev = prev['Close']
    change = (price_now - price_prev) / price_prev * 100

    key = f"{ticker}_trend"
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    signal_type = None

    # === Primary Signal Logic ===
    if ticker in COMMODITIES + ETFS:
        if change > 1.0 and rsi > 55 and macd > signal:
            signal_type = "STRONG BUY"
        elif change < -1.5 and rsi < 45 and macd < signal:
            signal_type = "STRONG SELL"
    elif ticker in STOCKS:
        if change > 1.5 and rsi > 60 and macd > signal:
            signal_type = "STRONG BUY"
        elif change < -2.0 and rsi < 40 and macd < signal:
            signal_type = "STRONG SELL"

    if key in signal_log and signal_log[key]['type'] == signal_type:
        if signal_type:
            carry_msg = f"[{ticker}] CONTINUED {signal_type} | Trend persistent | Reconfirming @ {price_now:.2f} | {timestamp}"
            send_telegram_alert(carry_msg)
            return carry_msg

    if signal_type:
        signal_msg = f"[{ticker}] {signal_type} | RSI: {rsi:.1f} | MACD: {macd:.2f} | Price: {price_now:.2f}"
        send_telegram_alert(signal_msg)
        signal_log[key] = {"type": signal_type, "price": price_now, "time": timestamp}

        # === Option Forecast ===
        if ticker in COMMODITIES:
            direction = "CALL" if "BUY" in signal_type else "PUT"
            entry = price_now
            target = price_now * (1.03 if direction == "CALL" else 0.97)
            stop = price_now * (0.985 if direction == "CALL" else 1.015)
            best_entry = price_now * (0.996 if direction == "CALL" else 1.004)
            option_msg = f"[{ticker}] {direction} OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f} | Best Entry: {best_entry:.2f}"
            send_telegram_alert(option_msg)
            return option_msg

        return signal_msg

    # === Apex & Reversal Signals ===
    if df['Hist'].iloc[-2] > df['Hist'].iloc[-1] and df['MACD'].iloc[-1] > 0:
        send_telegram_alert(f"[{ticker}] MACD Histogram Fading | Bull Momentum Weakening")

    if df['Hist'].iloc[-2] < df['Hist'].iloc[-1] and df['MACD'].iloc[-1] < 0:
        send_telegram_alert(f"[{ticker}] MACD Histogram Fading | Bear Momentum Weakening")

    if df['BBWidth'].iloc[-1] < df['BBWidthNorm'].iloc[-1] * 0.75:
        send_telegram_alert(f"[{ticker}] VOLATILITY SQUEEZE | Tight range forming | Big move likely soon")

    if (df['Close'].iloc[-1] > df['Close'].iloc[-2] and df['RSI'].iloc[-1] < df['RSI'].iloc[-2]):
        send_telegram_alert(f"[{ticker}] BEARISH DIVERGENCE | Price rising, RSI falling")

    if (df['Close'].iloc[-1] < df['Close'].iloc[-2] and df['RSI'].iloc[-1] > df['RSI'].iloc[-2]):
        send_telegram_alert(f"[{ticker}] BULLISH DIVERGENCE | Price falling, RSI rising")

    return None

def main():
    print("Running Alladin Ultimate Apex...")
    alerts = []
    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market closed. Skipping.")
            continue
        df = fetch_data(ticker)
        if df is None:
            print(f"[{ticker}] Data error.")
            continue
        result = evaluate_signal(df, ticker)
        if result:
            alerts.append(result)

    if alerts:
        print("\n--- SIGNALS ---")
        for alert in alerts:
            print(alert)
    else:
        print("No signals triggered.")

if __name__ == "__main__":
    main()
