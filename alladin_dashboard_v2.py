# Alladin Dashboard v2 — Now with Intraday Spike Reversal Alerts + Dynamic Mode
# Tracks 5-minute candle spikes for high-volatility tickers (DOT, DJT, BTC, WOLF, CL=F, NG=F)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        except Exception as e:
            print("Telegram error:", e)

memory_file = "signal_memory.csv"
if os.path.exists(memory_file):
    memory_log = pd.read_csv(memory_file)
else:
    memory_log = pd.DataFrame(columns=["Ticker", "Signal", "Date"])

tickers = [
    "NVDA", "PLTR", "SMCI", "LMT", "NOC", "CCJ", "ASML", "PM", "NEM", "T", "CVS",
    "CL=F", "BZ=F", "GC=F", "NG=F", "HG=F", "SPY", "QQQ", "XLE", "XLF", "XLK",
    "ARKK", "GDX", "BTC-USD", "ETH-USD", "AVIX", "DJT", "WOLF", "DOT-USD"
]

# Download recent 10-day daily data
end = datetime.now()
start = end - timedelta(days=10)
data = yf.download(tickers, start=start, end=end)

if data.empty:
    send_telegram_alert("Alladin Error: No market data.")
    exit()

if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.levels[0]:
        data = data["Adj Close"]
    elif "Close" in data.columns.levels[0]:
        data = data["Close"]
    else:
        send_telegram_alert("Alladin Error: No valid pricing data found.")
        exit()
elif "Adj Close" in data.columns:
    pass
elif "Close" in data.columns:
    pass
else:
    send_telegram_alert("Alladin Error: No pricing columns found.")
    exit()

data = data.dropna(axis=1, thresh=len(data) - 2)
returns = data.pct_change().dropna()
weekly_returns = returns.sum() * 100
volatility = returns.std() * 100
last_prices = data.iloc[-1]
trend = data.diff().iloc[-3:].sum()

# Determine market mode
avg_vol = volatility.mean()
market_mode = "STABLE" if avg_vol < 4 else "VOLATILE"
score_threshold = 3 if market_mode == "STABLE" else 2

insider_bias = {"WOLF": "SELL", "DJT": "BUY", "NVDA": "BUY", "PLTR": "SELL"}
sentiment_bias = {"WOLF": "NEGATIVE", "DJT": "POSITIVE", "NVDA": "POSITIVE", "PLTR": "NEGATIVE"}

def pattern_recognition(series):
    if len(series) < 6:
        return "NEUTRAL"
    s = series.iloc[-6:]
    if s.iloc[0] > s.iloc[1] > s.iloc[2] and s.iloc[3] < s.iloc[4] < s.iloc[5]:
        return "BUY"
    if s.iloc[0] < s.iloc[1] < s.iloc[2] and s.iloc[3] > s.iloc[4] > s.iloc[5]:
        return "SELL"
    return "NEUTRAL"

patterns = {t: pattern_recognition(data[t].dropna()) for t in data.columns}

dynamic_strategies = {}
for t in data.columns:
    if abs(trend[t]) > 0.5 and volatility[t] < 4 and abs(weekly_returns[t]) > 1.5:
        dynamic_strategies[t] = "Long-Term"
    elif volatility[t] > 4 or abs(trend[t]) > 0.3:
        dynamic_strategies[t] = "Short-Term"
    else:
        dynamic_strategies[t] = "Neutral"

def signal_logic(row):
    score = 0
    if row["Trend"] == "RISING" and row["Weekly Return (%)"] > 3:
        score += 1
    if row["Trend"] == "FALLING" and row["Weekly Return (%)"] < -3:
        score += 1
    if row["Insider Bias"] == "BUY": score += 1
    if row["Insider Bias"] == "SELL": score -= 1
    if row["Pattern"] == "BUY": score += 1
    if row["Pattern"] == "SELL": score -= 1
    if row["Sentiment"] == "POSITIVE": score += 1
    if row["Sentiment"] == "NEGATIVE": score -= 1
    if score >= score_threshold:
        confidence = "STRONG" if score >= 3 else "MODERATE"
        return f"{confidence} BUY" if score > 0 else f"{confidence} SELL", score
    return "NEUTRAL", score

df = pd.DataFrame({
    "Weekly Return (%)": weekly_returns.round(2),
    "Volatility (%)": volatility.round(2),
    "Trend": ["RISING" if trend[t] > 0 else "FALLING" if trend[t] < 0 else "STABLE" for t in data.columns],
    "Current Price": last_prices.round(2),
    "Strategy": [dynamic_strategies[t] for t in data.columns],
    "Insider Bias": [insider_bias.get(t, "NEUTRAL") for t in data.columns],
    "Pattern": [patterns.get(t, "NEUTRAL") for t in data.columns],
    "Sentiment": [sentiment_bias.get(t, "NEUTRAL") for t in data.columns]
})
df[["Signal", "Confidence"]] = df.apply(lambda row: pd.Series(signal_logic(row)), axis=1)

today = datetime.now().strftime("%Y-%m-%d")
new_signals = df[df["Signal"].str.contains("BUY|SELL")]
new_signals = new_signals[~new_signals.index.isin(memory_log[memory_log["Date"] == today]["Ticker"])]

def format_alert(ticker, row):
    return f"{ticker} | {row['Signal']} | Return: {row['Weekly Return (%)']}% | Vol: {row['Volatility (%)']}% | Trend: {row['Trend']}"

if new_signals.empty:
    send_telegram_alert(f"Alladin [{market_mode} Mode]: No strong signals.")
else:
    alerts = [f"Alladin [{market_mode} Mode] Signals:"]
    for ticker, row in new_signals.iterrows():
        alerts.append(format_alert(ticker, row))
        memory_log = pd.concat([memory_log, pd.DataFrame([[ticker, row["Signal"], today]], columns=["Ticker", "Signal", "Date"])])
    send_telegram_alert("\n".join(alerts))

# --- Signal Reversal Detection ---
reversal_alerts = []
if os.path.exists(memory_file):
    prev_signals = pd.read_csv(memory_file)
    merged = pd.merge(df.reset_index(), prev_signals, left_on="index", right_on="Ticker", how="inner")
    for _, row in merged.iterrows():
        prev = row["Signal_y"]
        curr = row["Signal_x"]
        if prev != "NEUTRAL" and curr != "NEUTRAL" and prev != curr:
            reversal_alerts.append(f"REVERSAL DETECTED: {row['Ticker']} switched from {prev} to {curr}")
    if reversal_alerts:
        send_telegram_alert("\n".join(reversal_alerts))

# --- Intraday Spike Reversal Logic (5-min candle) ---
spike_tickers = ["DOT-USD", "DJT", "BTC-USD", "WOLF", "CL=F", "NG=F"]
now = datetime.utcnow()
start_intraday = now - timedelta(minutes=60)
intraday_data = yf.download(spike_tickers, start=start_intraday, interval="5m")

if isinstance(intraday_data.columns, pd.MultiIndex):
    intraday_data = intraday_data["Close"]

spike_messages = []
for ticker in spike_tickers:
    if ticker not in intraday_data.columns:
        continue
    prices = intraday_data[ticker].dropna()
    if len(prices) >= 4:
        p_start = prices.iloc[0]
        p_low = prices.min()
        p_high = prices.max()
        p_now = prices.iloc[-1]
        drop = round((p_low - p_start) / p_start * 100, 2)
        rise = round((p_high - p_low) / p_low * 100, 2)
        rebound = round((p_now - p_low) / p_low * 100, 2)
        if abs(drop) >= 3 and rebound >= 3:
            spike_messages.append(f"{ticker} Intraday Reversal: Drop {drop}%, Rebound {rebound}%, High Swing {rise}% (5-min)")
if spike_messages:
    send_telegram_alert("Intraday Reversals (5-min):\n" + "\n".join(spike_messages))

memory_log.to_csv(memory_file, index=False)
df.to_csv("alladin_dashboard_v2.csv")
print("Alladin Dashboard saved.")
