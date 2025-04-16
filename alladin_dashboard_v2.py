# === Alladin Ultimate V2 with Adaptive Entry Targets ===
# A next-gen financial intelligence system with strike targets

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import pytz
import telegram

# === Telegram Setup ===
TELEGRAM_ENABLED = True
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_ENABLED else None

# === Ticker Classification ===
COMMODITIES = ['CL=F', 'NG=F', 'ETH-USD', 'BTC-USD']
STOCKS = ['DJT', 'WOLF', 'LMT', 'AAPL', 'TSLA', 'DOT']
ETFS = ['SPY', 'IVV']
TICKERS = COMMODITIES + STOCKS + ETFS
VIX_TICKER = '^VIX'

# === Strategy Configuration ===
def classify_strategy(ticker):
    if ticker in COMMODITIES:
        return 'momentum'
    elif ticker in STOCKS:
        return 'news-driven'
    elif ticker in ETFS:
        return 'macro-trend'
    return 'neutral'

# === Market Hours ===
def market_is_open(ticker):
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    wd, hour, minute = now.weekday(), now.hour, now.minute
    if wd >= 5: return False
    if ticker in STOCKS + ETFS:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)
    return True

# === Fetch VIX ===
def fetch_vix():
    try:
        vix = yf.download(VIX_TICKER, period='1d', interval='5m', progress=False)
        return vix['Close'].iloc[-1] if not vix.empty else None
    except: return None

# === Indicators ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    macd = series.ewm(span=12).mean() - series.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    return macd, signal

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# === Fetch Data ===
def fetch_data(ticker):
    df = yf.download(ticker, interval='15m', period='2d', progress=False)
    if df.empty or len(df) < 30:
        print(f"[{ticker}] No data or insufficient rows.")
        return None
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['ATR'] = compute_atr(df)
    df.dropna(inplace=True)
    return df

# === Confidence Scoring ===
def compute_confidence(df):
    last = df.iloc[-1]
    macd_diff = last['MACD'] - last['Signal']
    rsi_score = max(0, min(100, (last['RSI'] - 50) * 2))
    macd_score = 25 if macd_diff > 0 else -25
    return int(50 + rsi_score/2 + macd_score)

# === Signal Generator with Entry/Target ===
def generate_signal(ticker, df, vix):
    if df is None or len(df) < 3:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
    strat = classify_strategy(ticker)
    confidence = compute_confidence(df)
    atr = last['ATR'] if 'ATR' in last and not pd.isna(last['ATR']) else 0.01

    sentiment = 'NEUTRAL'
    if vix:
        if vix > 25: sentiment = 'CAUTION'
        elif vix < 15: sentiment = 'BULLISH'

    entry = round(float(last['Close']), 2)
    target_up = round(entry + atr * 1.2, 2)
    target_down = round(entry - atr * 1.2, 2)

    if strat == 'momentum':
        if change > 1.0 and last['MACD'] > last['Signal']:
            return f"{ticker}: CALL @ {entry} | Target: {target_up} | Stop: {target_down} | Conf: {confidence}% | {sentiment}"
        elif change < -1.0 and last['MACD'] < last['Signal']:
            return f"{ticker}: PUT @ {entry} | Target: {target_down} | Stop: {target_up} | Conf: {confidence}% | {sentiment}"

    if strat == 'news-driven' and confidence >= 70:
        return f"{ticker}: STRONG SIGNAL @ {entry} | Conf: {confidence}% | {sentiment}"

    if strat == 'macro-trend' and abs(change) > 0.75:
        return f"{ticker}: ETF MOVE {change:.2f}% @ {entry} | Conf: {confidence}% | {sentiment}"

    return None

# === Telegram Alert ===
def send_telegram_alert(message):
    try:
        if TELEGRAM_ENABLED and bot:
            bot.send_message(chat_ 
                             id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Telegram error: {e}")

# === Main ===
def main():
    print("Running Alladin Ultimate V2 with Targets")
    vix = fetch_vix()
    for ticker in TICKERS:
        if not market_is_open(ticker):
            continue
        df = fetch_data(ticker)
        signal = generate_signal(ticker, df, vix)
        if signal:
            send_telegram_alert(signal)
            print(signal)

if __name__ == "__main__":
    main()
