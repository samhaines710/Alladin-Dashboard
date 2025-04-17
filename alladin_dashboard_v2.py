#!/usr/bin/env python3
import os
import datetime as dt
import pytz
import logging

import yfinance as yf
import pandas as pd
import numpy as np

# === Telegram Setup ===
try:
    from telegram import Bot
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
    TELEGRAM_ENABLED = bool(bot)
except ImportError:
    bot = None
    TELEGRAM_ENABLED = False

# === Logging ===
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)
# CSV log for P&L tracking
LOG_COLUMNS = ["timestamp","ticker","signal","entry","target","stop","result"]
if not os.path.exists("log.csv"):
    pd.DataFrame(columns=LOG_COLUMNS).to_csv("log.csv", index=False)

# === Ticker Categories ===
COMMODITIES = ["CL=F","NG=F","ETH-USD","BTC-USD"]
STOCKS      = ["DJT","WOLF","LMT","AAPL","TSLA","DOT"]
ETFS        = ["SPY","IVV"]
TICKERS     = COMMODITIES + STOCKS + ETFS

# === News Overrides (simple preset) ===
NEWS_SENTIMENT = {
    "DJT":"Bearish","WOLF":"Neutral","LMT":"Neutral","AAPL":"Bullish",
    "TSLA":"Neutral","DOT":"Neutral","SPY":"Neutral","IVV":"Neutral"
}

def get_news_sentiment(ticker: str) -> str:
    return NEWS_SENTIMENT.get(ticker, "Neutral")

# === Market Hours (RSA) ===
def market_is_open(ticker: str) -> bool:
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    now = utc_now.astimezone(pytz.timezone("Africa/Johannesburg"))
    if now.weekday() >= 5:
        return False
    if ticker in STOCKS or ticker in ETFS:
        # 15:30–22:00 SAST
        return (now.hour == 15 and now.minute >= 30) or (16 <= now.hour < 22)
    return True  # commodities & crypto 24/7

# === Indicators ===
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def compute_macd(prices: pd.Series, short=12, long=26, sig=9):
    e1 = prices.ewm(span=short, adjust=False).mean()
    e2 = prices.ewm(span=long,  adjust=False).mean()
    macd = e1 - e2
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["range1"] = d["High"] - d["Low"]
    d["range2"] = (d["High"] - d["prev_close"]).abs()
    d["range3"] = (d["Low"] - d["prev_close"]).abs()
    d["tr"] = d[["range1","range2","range3"]].max(axis=1)
    return d["tr"].rolling(period).mean()

# === Data Fetching ===
def fetch_data(ticker: str, interval="15m", period="2d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False)
        if df is None or df.empty or len(df) < 30:
            logging.info(f"[{ticker}] No data or insufficient rows.")
            return None
        # flatten columns if multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip() for c in df.columns.values]
            rename_map = {}
            for c in df.columns:
                lc = c.lower()
                if "open" in lc:   rename_map[c] = "Open"
                if "high" in lc:   rename_map[c] = "High"
                if "low" in lc:    rename_map[c] = "Low"
                if "close" in lc and "adj" not in lc:
                    rename_map[c] = "Close"
                if "volume" in lc: rename_map[c] = "Volume"
            df.rename(columns=rename_map, inplace=True)

        if not {"High","Low","Close"}.issubset(df.columns):
            logging.warning(f"[{ticker}] Missing columns {set(['High','Low','Close'])-set(df.columns)}")
            return None

        df["RSI"] = compute_rsi(df["Close"])
        macd, signal = compute_macd(df["Close"])
        df["MACD"], df["Signal"] = macd, signal
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"[{ticker}] fetch_data error: {e}")
        return None

# === Telegram ===
def send_msg(text: str):
    logging.info("ALERT: " + text)
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=text)
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

# === Options Logic ===
def option_signal(ticker: str, df: pd.DataFrame, sig: str):
    if ticker not in ["CL=F","NG=F"]:
        return None
    price = float(df["Close"].iloc[-1])
    atr_val = compute_atr(df).iloc[-1]
    # dynamic thresholds
    tgt_pct, stp_pct = (0.02, 0.01)
    if atr_val/price > 0.01:
        tgt_pct, stp_pct = (0.03, 0.015)

    if sig in ("STRONG BUY","BUY"):
        entry = price
        tgt = price*(1+tgt_pct)
        stp = price*(1-stp_pct)
        return f"{ticker}: ▶️ CALL | E={entry:.2f} T={tgt:.2f} S={stp:.2f}"
    if sig in ("STRONG SELL","SELL"):
        entry = price
        tgt = price*(1-tgt_pct)
        stp = price*(1+tgt_pct)
        return f"{ticker}: ◀️ PUT  | E={entry:.2f} T={tgt:.2f} S={stp:.2f}"
    return None

# === Signal Evaluation ===
def evaluate_signals(df: pd.DataFrame, ticker: str):
    try:
        # require at least 5 rows
        if df is None or len(df)<5:
            return None

        close_now  = df["Close"].iloc[-1]
        close_prev = df["Close"].iloc[-2]
        pct_change = (close_now-close_prev)/close_prev*100
        rsi        = df["RSI"].iloc[-1]
        macd       = df["MACD"].iloc[-1]
        signal_ln  = df["Signal"].iloc[-1]

        tag = None
        reasons = []

        # thresholds by category
        if ticker in COMMODITIES+ETFS:
            if pct_change>1.0 and rsi>55 and macd>signal_ln:
                tag="STRONG BUY"
            elif pct_change<-1.5 and rsi<45 and macd<signal_ln:
                tag="STRONG SELL"
            elif macd>signal_ln and pct_change>0.5:
                tag="BUY"
            elif macd<signal_ln and pct_change<-0.5:
                tag="SELL"

        else:  # STOCKS
            if pct_change>1.5 and rsi>60 and macd>signal_ln:
                tag="STRONG BUY"
            elif pct_change<-2.0 and rsi<40 and macd<signal_ln:
                tag="STRONG SELL"
            elif macd>signal_ln and pct_change>1.0:
                tag="BUY"
            elif macd<signal_ln and pct_change<-1.0:
                tag="SELL"
            # news override
            sent = get_news_sentiment(ticker)
            if sent=="Bearish" and tag in ("STRONG BUY","BUY"):
                tag="SUSPEND BUY"
            if sent=="Bullish" and tag in ("STRONG SELL","SELL"):
                tag="SUSPEND SELL"

        # collect reasons
        if tag:
            reasons.append(f"{pct_change:+.2f}%")
            reasons.append(f"RSI {rsi:.1f}")
            reasons.append("MACD→" + ("▲" if macd>signal_ln else "▼"))

        # reversal detection
        # look back two bars
        prev3_macd, prev3_sig = df["MACD"].iloc[-4], df["Signal"].iloc[-4]
        prev2_macd, prev2_sig = df["MACD"].iloc[-3], df["Signal"].iloc[-3]
        if prev3_macd<prev3_sig and prev2_macd<prev2_sig and macd>signal_ln:
            send_msg(f"{ticker}: 🔄 REVERSAL UP | Watch @ {close_now:.2f}")
        elif prev3_macd>prev3_sig and prev2_macd>prev2_sig and macd<signal_ln:
            send_msg(f"{ticker}: 🔄 REVERSAL DN | Watch @ {close_now:.2f}")

        # main alert
        if tag:
            msg = f"{ticker}: {tag} | "+ " | ".join(reasons)
            send_msg(msg)

            # options if commodity
            opt = option_signal(ticker, df, tag)
            if opt:
                send_msg(opt)
                log_row = {
                    "timestamp": dt.datetime.now().isoformat(),
                    "ticker": ticker,
                    "signal": tag,
                    "entry": df["Close"].iloc[-1],
                    "target": None,
                    "stop": None,
                    "result": None
                }
                pd.DataFrame([log_row])[LOG_COLUMNS].to_csv(
                    "log.csv", mode="a", header=False, index=False
                )
            return msg

    except Exception as e:
        logging.error(f"[{ticker}] Signal error: {e}")
    return None

# === Main ===
def main():
    rsa_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)\
               .astimezone(pytz.timezone("Africa/Johannesburg"))
    logging.info(f"Running Alladin at RSA {rsa_now:%Y-%m-%d %H:%M:%S}")

    for tk in TICKERS:
        if not market_is_open(tk):
            logging.debug(f"[{tk}] Market closed.")
            continue
        df = fetch_data(tk)
        evaluate_signals(df, tk)

if __name__=="__main__":
    main()
