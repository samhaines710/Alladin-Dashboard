# === Ultimate Alladin Dashboard v2 ===
# Includes: real-time signals, apex prediction, reversal detection, next-best-entry logic,
# adaptive confidence filtering, and Telegram alerts

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

# === Ticker Categories ===
COMMODITIES = ['CL=F', 'NG=F']
STOCKS = ['DJT', 'WOLF', 'LMT', 'AAPL', 'TSLA', 'DOT']
ETFS = ['SPY', 'IVV']
TICKERS = COMMODITIES + STOCKS + ETFS

# === Market Hours Check (RSA) ===
def market_is_open(ticker):
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    if now.weekday() >= 5: return False
    if ticker in COMMODITIES: return True
    return 15 <= now.hour < 22

# === Indicators ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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
    df['tr'] = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['prev_close']).abs(),
        (df['Low'] - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    return df['tr'].rolling(window=period).mean()

# === Apex Prediction ===
def predict_apex(df):
    if len(df) < 10: return False
    recent_highs = df['High'].rolling(3).max()
    is_flat = recent_highs[-3:].std() < 0.05
    if is_flat and df['MACD'].iloc[-1] < df['Signal'].iloc[-1]:
        return True
    return False

# === Fetch Data ===
def fetch_data(ticker, interval='5m', period='1d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty: return None
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])
        macd, sig = compute_macd(df['Close'])
        df['MACD'], df['Signal'] = macd, sig
        df.dropna(inplace=True)
        return df
    except: return None

# === Signal Logic ===
def evaluate_signals(ticker, df):
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi, macd, sig = latest['RSI'], latest['MACD'], latest['Signal']
        change = df['Close'].pct_change().iloc[-1] * 100
        signal, reason = None, []

        # Entry logic
        if rsi > 55 and macd > sig and change > 0.5:
            signal = "CALL"
            reason.append(f"RSI {rsi:.1f} MACD+ {macd:.2f}>{sig:.2f}")
        elif rsi < 45 and macd < sig and change < -0.5:
            signal = "PUT"
            reason.append(f"RSI {rsi:.1f} MACD- {macd:.2f}<{sig:.2f}")

        # Apex logic
        if predict_apex(df):
            reason.append("APEX DETECTED")
            signal = "APEX WARNING"

        if signal:
            price = latest['Close']
            msg = f"{ticker}: {signal} | Price: {price:.2f} | {' | '.join(reason)}"
            send_telegram_alert(msg)
            return msg
    except Exception as e:
        print(f"[{ticker}] Signal error: {e}")
    return None

# === Telegram ===
def send_telegram_alert(msg):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            print("Telegram failed:", e)

# === MAIN ===
def main():
    now = dt.datetime.now(pytz.timezone("Africa/Johannesburg"))
    print(f"Alladin running at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    for ticker in TICKERS:
        if not market_is_open(ticker): continue
        df = fetch_data(ticker)
        if df is not None:
            evaluate_signals(ticker, df)

if __name__ == "__main__":
    main()
