import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests

# Telegram Config
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        except Exception as e:
            print("Telegram error:", e)

# Load signal memory
memory_file = "signal_memory.csv"
if os.path.exists(memory_file):
    memory_log = pd.read_csv(memory_file)
else:
    memory_log = pd.DataFrame(columns=["Ticker", "Signal", "Date"])

tickers = [
    "NVDA", "PLTR", "SMCI", "LMT", "NOC", "CCJ", "ASML", "PM", "NEM", "T", "CVS",
    "CL=F", "BZ=F", "GC=F", "NG=F", "HG=F", "SPY", "QQQ", "XLE", "XLF", "XLK",
    "ARKK", "GDX", "BTC-USD", "ETH-USD", "AVIX", "DJT", "WOLF"
]

# Download data
end = datetime.now()
start = end - timedelta(days=10)
data = yf.download(tickers, start=start, end=end)

# Handle missing or bad data
if data.empty:
    send_telegram_alert("Alladin Error: No data received from YFinance.")
    exit()

# Handle 'Adj Close' or 'Close' robustly
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.levels[0]:
        data = data['Adj Close']
    elif 'Close' in data.columns.levels[0]:
        data = data['Close']
    else:
        send_telegram_alert("Alladin Error: No valid price columns found in data.")
        exit()
elif 'Adj Close' in data.columns:
    pass
elif 'Close' in data.columns:
    pass
else:
    send_telegram_alert("Alladin Error: No 'Adj Close' or 'Close' in flat data.")
    exit()

data = data.dropna(axis=1, thresh=len(data) - 2)
returns = data.pct_change().dropna()
weekly_returns = returns.sum() * 100
volatility = returns.std() * 100
last_prices = data.iloc[-1]
trend = data.diff().iloc[-3:].sum()

# Insider and sentiment placeholders
insider_bias = {"WOLF": "SELL", "DJT": "BUY", "NVDA": "BUY", "PLTR": "SELL"}
sentiment_bias = {"WOLF": "NEGATIVE", "DJT": "POSITIVE", "NVDA": "POSITIVE", "PLTR": "NEGATIVE"}

# Pattern recognition
def pattern_recognition(series):
    if len(series) < 6:
        return "NEUTRAL"
    if series.iloc[-6] > series.iloc[-5] > series.iloc[-4] and series.iloc[-3] < series.iloc[-2] < series.iloc[-1]:
        return "BUY"
    if series.iloc[-6] < series.iloc[-5] < series.iloc[-4] and series.iloc[-3] > series.iloc[-2] > series.iloc[-1]:
        return "SELL"
    return "NEUTRAL"

patterns = {t: pattern_recognition(data[t].dropna()) for t in data.columns}

# Strategy classifier
dynamic_strategies = {}
for t in data.columns:
    if abs(trend[t]) > 0.5 and volatility[t] < 4 and abs(weekly_returns[t]) > 1.5:
        dynamic_strategies[t] = "Long-Term"
    elif volatility[t] > 4 or abs(trend[t]) > 0.3:
        dynamic_strategies[t] = "Short-Term"
    else:
        dynamic_strategies[t] = "Neutral"

# Signal logic
def signal_logic(row):
    score = 0
    if row["Trend"] == "RISING" and row["Weekly Return (%)"] > 3:
        score += 1
    if row["Trend"] == "FALLING" and row["Weekly Return (%)"] < -3:
        score += 1
    if row["Insider Bias"] == "BUY":
        score += 1
    elif row["Insider Bias"] == "SELL":
        score -= 1
    if row["Pattern"] == "BUY":
        score += 1
    elif row["Pattern"] == "SELL":
        score -= 1
    if row["Sentiment"] == "POSITIVE":
        score += 1
    elif row["Sentiment"] == "NEGATIVE":
        score -= 1
    if score >= 3:
        return "STRONG BUY", score
    elif score <= -3:
        return "STRONG SELL", score
    return "NEUTRAL", score

# Build dataframe
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

# Filter new signals
today = datetime.now().strftime("%Y-%m-%d")
new_signals = df[df["Signal"].isin(["STRONG BUY", "STRONG SELL"])]
new_signals = new_signals[~new_signals.index.isin(memory_log[memory_log["Date"] == today]["Ticker"])]

def format_alert(ticker, row):
    return f"""
{ticker} | {row['Signal']} | Return: {row['Weekly Return (%)']}%
Volatility: {row['Volatility (%)']}% | Trend: {row['Trend']}
"""

if new_signals.empty:
    send_telegram_alert("Alladin: No STRONG BUY/SELL signals right now.")
else:
    alerts = ["ALERTS:"]
    for ticker, row in new_signals.iterrows():
        alerts.append(format_alert(ticker, row))
        memory_log = pd.concat([memory_log, pd.DataFrame([[ticker, row["Signal"], today]], columns=["Ticker", "Signal", "Date"])])
    send_telegram_alert("\n".join(alerts))

# Save memory and output
memory_log.to_csv(memory_file, index=False)
df.to_csv("alladin_dashboard_v2.csv")
