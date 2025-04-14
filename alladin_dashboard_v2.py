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
    "NVDA", "PLTR", "SMCI", "LMT", "NOC", "CCJ", "ASML", "PM",
    "NEM", "T", "CVS", "CL=F", "BZ=F", "GC=F", "NG=F", "HG=F",
    "SPY", "QQQ", "XLE", "XLF", "XLK", "ARKK", "GDX", "BTC-USD",
    "ETH-USD", "AVIX", "DJT", "WOLF"
]

# Fetch price data
end = datetime.now()
start = end - timedelta(days=10)
data = yf.download(tickers, start=start, end=end)

# Check for empty or failed download
if data.empty or len(data.columns) == 0:
    print("No data fetched — exiting safely.")
    send_telegram_alert("Alladin Error: No data available from yfinance.")
    exit()

# Handle Adjusted Close logic
if isinstance(data.columns, pd.MultiIndex) and "Adj Close" in data.columns.levels[0]:
    data = data["Adj Close"]
elif "Adj Close" in data.columns:
    data = data["Adj Close"]
else:
    print("No 'Adj Close' found. Exiting...")
    send_telegram_alert("Alladin Error: 'Adj Close' missing in data.")
    exit()

data = data.dropna(axis=1, thresh=len(data) - 2)
returns = data.pct_change().dropna()
weekly_returns = returns.sum() * 100
volatility = returns.std() * 100
last_prices = data.iloc[-1]
trend = data.diff().iloc[-3:].sum()

# Debug
print("Data columns:", list(data.columns))
print("Length of df inputs:", len(last_prices))

# Bias Sources
insider_bias = {"WOLF": "SELL", "DJT": "BUY", "NVDA": "BUY", "PLTR": "SELL"}
sentiment_bias = {"WOLF": "NEGATIVE", "DJT": "POSITIVE", "NVDA": "POSITIVE", "PLTR": "NEGATIVE"}
thresholds = {"default_buy": 3, "default_sell": -3}

# Pattern detection
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

# Strategy tagging
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
    if row["Trend"] == "RISING" and row["Weekly Return (%)"] > thresholds["default_buy"]:
        score += 1
    if row["Trend"] == "FALLING" and row["Weekly Return (%)"] < thresholds["default_sell"]:
        score += 1
    score += {"BUY": 1, "SELL": -1}.get(row["Insider Bias"], 0)
    score += {"BUY": 1, "SELL": -1}.get(row["Pattern"], 0)
    score += {"POSITIVE": 1, "NEGATIVE": -1}.get(row["Sentiment"], 0)

    if score >= 3:
        return "STRONG BUY", score
    elif score <= -3:
        return "STRONG SELL", score
    return "NEUTRAL", score

# Build main DataFrame
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

signal_results = df.apply(signal_logic, axis=1).tolist()
if signal_results:
    df["Signal"], df["Confidence"] = zip(*signal_results)
else:
    df["Signal"], df["Confidence"] = "NEUTRAL", 0
    print("No valid signal results to unpack.")

# New trades only
today = datetime.now().strftime("%Y-%m-%d")
new_signals = df[df["Signal"].isin(["STRONG BUY", "STRONG SELL"])]
new_signals = new_signals[~new_signals.index.isin(memory_log[memory_log["Date"] == today]["Ticker"])]

# Alert message format
def format_alert(ticker, row):
    price = row["Current Price"]
    trade_type = "CALL" if row["Signal"] == "STRONG BUY" else "PUT"
    strike = round(price, 2)
    stop_loss = round(price * 0.975, 2) if trade_type == "CALL" else round(price * 1.025, 2)
    tp2 = round(price * 1.03, 2) if trade_type == "CALL" else round(price * 0.97, 2)
    return f"""
{ticker} TRADE ALERT: {row['Signal']}
----------------------------------------
Price: ${price} → Target: ${tp2}
Trend: {row['Trend']} | Strategy: {row['Strategy']}
Volatility: {row['Volatility (%)']}% | Return: {row['Weekly Return (%)']}%
Pattern: {row['Pattern']} | Insider: {row['Insider Bias']} | Sentiment: {row['Sentiment']}
Confidence Score: {row['Confidence']} / 5
Option Strategy: {trade_type} @ ${strike} | SL: ${stop_loss} | Exp: 1W
"""

for ticker, row in new_signals.iterrows():
    send_telegram_alert(format_alert(ticker, row))
    memory_log = pd.concat([memory_log, pd.DataFrame([[ticker, row["Signal"], today]], columns=["Ticker", "Signal", "Date"])])

memory_log.to_csv(memory_file, index=False)

# OIL/NG ENTRY ZONES
zone_window = 10
zone_data = yf.download(["CL=F", "NG=F"], start=datetime.now() - timedelta(days=15), end=datetime.now())

if isinstance(zone_data.columns, pd.MultiIndex) and "Adj Close" in zone_data.columns.levels[0]:
    zone_data = zone_data["Adj Close"]
elif "Adj Close" in zone_data.columns:
    zone_data = zone_data["Adj Close"]
else:
    print("Zone 'Adj Close' missing.")
    send_telegram_alert("Oil/NG data error: 'Adj Close' not found.")
    zone_data = pd.DataFrame()

if not zone_data.empty:
    zone_info = zone_data[-zone_window:]
    mean_prices = zone_info.mean()
    price_std = zone_info.std()

    zones = {}
    for ticker in ["CL=F", "NG=F"]:
        mean = mean_prices[ticker]
        std = price_std[ticker]
        call_zone = (round(mean - 1.2 * std, 2), round(mean - 0.5 * std, 2))
        put_zone = (round(mean + 0.5 * std, 2), round(mean + 1.2 * std, 2))
        zones[ticker] = {"CALL": call_zone, "PUT": put_zone}

    oil_ng_alerts = []
    latest_prices = data.iloc[-1]

    for ticker in ["CL=F", "NG=F"]:
        price = latest_prices[ticker]
        call_low, call_high = zones[ticker]["CALL"]
        put_low, put_high = zones[ticker]["PUT"]

        if call_low <= price <= call_high:
            oil_ng_alerts.append(f"{ticker} CALL ENTRY ZONE: ${price} in ${call_low}-${call_high} | Mean: ${round(mean_prices[ticker],2)} ± {round(price_std[ticker],2)}")
        elif put_low <= price <= put_high:
            oil_ng_alerts.append(f"{ticker} PUT ENTRY ZONE: ${price} in ${put_low}-${put_high} | Mean: ${round(mean_prices[ticker],2)} ± {round(price_std[ticker],2)}")

    if oil_ng_alerts:
        send_telegram_alert("DYNAMIC OIL/NG OPTIONS ZONES:\n\n• " + "\n• ".join(oil_ng_alerts))
    else:
        send_telegram_alert("Oil/NG: Price outside optimal dynamic zones.")
else:
    print("Zone data unavailable.")

df.to_csv("alladin_dashboard_v2.csv")
print("Dashboard saved as alladin_dashboard_v2.csv")
