#!/usr/bin/env python3
import os, datetime as dt, pytz, logging
import yfinance as yf
import pandas as pd, numpy as np
try:
    from telegram import Bot
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
    TELEGRAM_ENABLED = bool(bot)
except ImportError:
    bot = None; TELEGRAM_ENABLED = False

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

COMMODITIES = ["CL=F","NG=F"]
CRYPTO      = ["ETH-USD","BTC-USD"]
STOCKS      = ["DJT","WOLF","LMT","AAPL","TSLA","DOT"]
ETFS        = ["SPY","IVV"]
TICKERS     = COMMODITIES + CRYPTO + STOCKS + ETFS

NEWS_SENTIMENT = {
    "DJT":"Bearish","WOLF":"Neutral","LMT":"Neutral","AAPL":"Bullish",
    "TSLA":"Neutral","DOT":"Neutral","SPY":"Neutral","IVV":"Neutral"
}
def get_news_sentiment(tk): return NEWS_SENTIMENT.get(tk,"Neutral")

def market_is_open(tk):
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)\
           .astimezone(pytz.timezone("Africa/Johannesburg"))
    if now.weekday()>=5: return False
    if tk in STOCKS+ETFS:
        return (now.hour==15 and now.minute>=30) or (16<=now.hour<22)
    return True

def compute_rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(period).mean() / dn.rolling(period).mean()
    return 100 - (100/(1+rs))

def compute_macd(s, short=12, long=26, signal=9):
    e1 = s.ewm(span=short, adjust=False).mean()
    e2 = s.ewm(span=long,  adjust=False).mean()
    macd = e1 - e2
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def compute_atr(df, period=14):
    d = df.copy()
    d["prev"] = d["Close"].shift(1)
    d["r1"] = d["High"] - d["Low"]
    d["r2"] = (d["High"] - d["prev"]).abs()
    d["r3"] = (d["Low"]  - d["prev"]).abs()
    tr = d[["r1","r2","r3"]].max(axis=1)
    return tr.rolling(period).mean()

def fetch_data(tk, interval="15m", period="2d"):
    try:
        df = yf.download(tk, interval=interval, period=period, progress=False)
        if df is None or len(df)<30:
            logging.info(f"[{tk}] insufficient data")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip() for c in df.columns.values]
            cmap = {}
            for c in df.columns:
                lc=c.lower()
                if "open" in lc:   cmap[c]="Open"
                if "high" in lc:   cmap[c]="High"
                if "low"  in lc:   cmap[c]="Low"
                if "close" in lc and "adj" not in lc: cmap[c]="Close"
                if "volume" in lc: cmap[c]="Volume"
            df.rename(columns=cmap,inplace=True)
        if not {"High","Low","Close"}.issubset(df.columns):
            logging.warning(f"[{tk}] missing required columns")
            return None
        df["RSI"] = compute_rsi(df["Close"])
        m,s = compute_macd(df["Close"])
        df["MACD"], df["Signal"] = m,s
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"[{tk}] fetch error: {e}")
        return None

def send_msg(txt):
    logging.info("ALERT: "+txt)
    if TELEGRAM_ENABLED and bot:
        try: bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=txt)
        except Exception as e: logging.error("tg failed: "+str(e))

def option_signal(tk, df, tag):
    if tk not in COMMODITIES: return None
    price = float(df["Close"].iloc[-1])
    atr   = compute_atr(df).iloc[-1]
    thr = atr/price
    # base 2×ATR for contract sizing
    if tag in ("BUY","STRONG BUY"):
        E = price; T=price*(1+2*thr); S=price*(1-thr)
        return f"{tk}: ▶️ CALL E={E:.2f} T={T:.2f} S={S:.2f}"
    if tag in ("SELL","STRONG SELL"):
        E = price; T=price*(1-2*thr); S=price*(1+thr)
        return f"{tk}: ◀️ PUT  E={E:.2f} T={T:.2f} S={S:.2f}"
    return None

def evaluate_signals(df, tk):
    try:
        if df is None or len(df)<5: return
        now_p  = df["Close"].iloc[-1]
        prev_p = df["Close"].iloc[-2]
        pct    = (now_p-prev_p)/prev_p*100
        rsi    = df["RSI"].iloc[-1]
        macd   = df["MACD"].iloc[-1]
        sig    = df["Signal"].iloc[-1]

        tag = None

        # commodities: 1×ATR bar move + MACD
        if tk in COMMODITIES:
            atr = compute_atr(df).iloc[-1]
            thr_pct = (atr/now_p)*100
            if thr_pct < 0.2: thr_pct = 0.2
            if pct > thr_pct and macd>sig:
                tag="BUY"
            elif pct < -thr_pct and macd<sig:
                tag="SELL"

        # crypto
        elif tk in CRYPTO:
            if pct>1.0 and rsi>55 and macd>sig: tag="STRONG BUY"
            elif pct<-1.5 and rsi<45 and macd<sig: tag="STRONG SELL"
            elif pct>0.5 and macd>sig: tag="BUY"
            elif pct<-0.5 and macd<sig: tag="SELL"

        # stocks & etfs
        else:
            if pct>1.5 and rsi>60 and macd>sig: tag="STRONG BUY"
            elif pct<-2.0 and rsi<40 and macd<sig: tag="STRONG SELL"
            elif pct>1.0 and macd>sig: tag="BUY"
            elif pct<-1.0 and macd<sig: tag="SELL"
            # news override
            news = get_news_sentiment(tk)
            if news=="Bearish" and tag in ("BUY","STRONG BUY"): tag="SUSPEND BUY"
            if news=="Bullish" and tag in ("SELL","STRONG SELL"): tag="SUSPEND SELL"

        # 2‑bar reversal
        m3,s3 = df["MACD"].iloc[-4], df["Signal"].iloc[-4]
        m2,s2 = df["MACD"].iloc[-3], df["Signal"].iloc[-3]
        if m3<s3 and m2<s2 and macd>sig:
            send_msg(f"{tk}: 🔄 REV UP | watch@{now_p:.2f}")
        if m3>s3 and m2>s2 and macd<sig:
            send_msg(f"{tk}: 🔄 REV DN | watch@{now_p:.2f}")

        if tag:
            send_msg(f"{tk}: {tag} | {pct:+.2f}% | RSI{rsi:.1f}")
            opt = option_signal(tk, df, tag)
            if opt: send_msg(opt)

    except Exception as e:
        logging.error(f"[{tk}] eval error: {e}")

def main():
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)\
           .astimezone(pytz.timezone("Africa/Johannesburg"))
    logging.info(f"Running Alladin @ RSA {now:%Y-%m-%d %H:%M:%S}")
    for tk in TICKERS:
        if not market_is_open(tk):
            logging.debug(f"[{tk}] closed.")
            continue
        df = fetch_data(tk)
        evaluate_signals(df, tk)

if __name__=="__main__":
    main()
