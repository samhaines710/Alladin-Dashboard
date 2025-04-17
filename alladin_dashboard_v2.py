#!/usr/bin/env python3
# Alladin Dashboard v2 – 5m Candles with High‑Signal Filters

import os
import datetime as dt
import logging
import pytz

import yfinance as yf
import pandas as pd
import numpy as np
try:
    from telegram import Bot
    TELEGRAM_ENABLED = True
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
except ImportError:
    TELEGRAM_ENABLED = False
    bot = None

# === Logging ===
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# === Tickers ===
COMMODITIES = ["CL=F","NG=F"]
CRYPTO      = ["ETH-USD","BTC-USD"]
STOCKS      = ["DJT","WOLF","LMT","AAPL","TSLA","DOT"]
ETFS        = ["SPY","IVV"]
TICKERS     = COMMODITIES + CRYPTO + STOCKS + ETFS

# === News Overrides ===
NEWS_SENTIMENT = {"WOLF":"Neutral","LMT":"Neutral","AAPL":"Bullish","TSLA":"Neutral","DOT":"Neutral","SPY":"Neutral","IVV":"Neutral"}
def get_news_sentiment(tk): return NEWS_SENTIMENT.get(tk, "Neutral")

# === Market Hours (RSA) ===
def market_is_open(tk):
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    if now.weekday()>=5: return False
    if tk in STOCKS+ETFS:
        return (now.hour==15 and now.minute>=30) or (16<=now.hour<22)
    return True

# === Indicators ===
def compute_rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(period).mean()/dn.rolling(period).mean()
    return 100 - (100/(1+rs))

def compute_macd(s, short=12, long=26, signal=9):
    e1 = s.ewm(span=short, adjust=False).mean()
    e2 = s.ewm(span=long,  adjust=False).mean()
    macd = e1 - e2
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def compute_atr(df, period=14):
    d = df.copy()
    d['prev'] = d['Close'].shift(1)
    tr = pd.concat([
        d['High']-d['Low'],
        (d['High']-d['prev']).abs(),
        (d['Low']-d['prev']).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# === Fetch Data ===
def fetch_data(tk, interval='5m', period='2d'):
    try:
        df = yf.download(tk, interval=interval, period=period, progress=False)
        if df is None or len(df)<30: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(c).strip() for c in df.columns.values]
            df.rename(columns=lambda x: x.split('_')[-1].capitalize(), inplace=True)
        if not {'High','Low','Close','Volume'}.issubset(df.columns): return None
        df['RSI']   = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])
        df['VWAP']  = (df['Close']*df['Volume']).cumsum()/df['Volume'].cumsum()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"[{tk}] fetch error: {e}")
        return None

# === Simple Signal (no filters) ===
def simple_signal(df, tk):
    now_p = df['Close'].iloc[-1]
    prev_p= df['Close'].iloc[-2]
    pct   = (now_p-prev_p)/prev_p*100
    rsi   = df['RSI'].iloc[-1]
    macd  = df['MACD'].iloc[-1]
    sig   = df['Signal'].iloc[-1]
    tag = None
    if tk in COMMODITIES:
        thr = (compute_atr(df).iloc[-1]/now_p)*100
        if thr<0.2: thr=0.2
        if pct>thr and macd>sig: tag='BUY'
        elif pct<-thr and macd<sig: tag='SELL'
    elif tk in CRYPTO or tk in STOCKS+ETFS:
        if pct>1.5 and rsi>55 and macd>sig: tag='STRONG BUY'
        elif pct<-1.5 and rsi<45 and macd<sig: tag='STRONG SELL'
        elif pct>0.5 and macd>sig: tag='BUY'
        elif pct<-0.5 and macd<sig: tag='SELL'
    return tag

# === Evaluate & Send Signals ===
def evaluate_signals(tk, df):
    try:
        # 1) Volume filter
        vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
        if df['Volume'].iloc[-1] < vol_ma: return
        # 2) Histogram acceleration
        hist = df['MACD'] - df['Signal']
        if hist.iloc[-1] <= hist.iloc[-2]: return
        # 3) 5m signal
        tag5 = simple_signal(df, tk)
        if not tag5: return
        # 4) 15m confirmation
        df15 = fetch_data(tk, interval='15m', period='2d')
        tag15 = simple_signal(df15, tk) if df15 is not None else None
        if tag15 != tag5: return
        # 5) VWAP filter
        last_close = df['Close'].iloc[-1]
        last_vwap  = df['VWAP'].iloc[-1]
        if tag5=='BUY' and last_close < last_vwap: return
        if tag5=='SELL' and last_close > last_vwap: return
        # 6) Send final alert
        send_msg(f"{tk}: {tag5} | {pct:+.2f}% | RSI{rsi:.1f}")
    except Exception as e:
        logging.error(f"[{tk}] eval error: {e}")

# === Telegram Helper ===
def send_msg(txt):
    logging.info("ALERT: " + txt)
    if TELEGRAM_ENABLED and bot:
        try: bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=txt)
        except Exception as e: logging.error(f"tg fail: {e}")

# === Main ===
def main():
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    logging.info(f"Running Alladin @ {now:%Y-%m-%d %H:%M:%S}")
    for tk in TICKERS:
        if not market_is_open(tk): continue
        df = fetch_data(tk)
        if df is not None: evaluate_signals(tk, df)

if __name__ == '__main__':
    main()
