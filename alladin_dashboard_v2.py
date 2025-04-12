import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

# Ticker categories
tickers = {
    "NVDA": "Tech", "PLTR": "Tech", "SMCI": "Tech/AI",
    "LMT": "Defense", "NOC": "Defense", "CCJ": "Energy/Nuclear",
    "ASML": "Semiconductors", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "NEM": "Materials/Gold", "T": "Telecom", "CVS": "Healthcare",
    "CL=F": "Crude Oil", "BZ=F": "Brent Oil", "GC=F": "Gold",
    "SI=F": "Silver", "NG=F": "Natural Gas", "HG=F": "Copper",
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "XLE": "Energy", "XLF": "Financials",
    "XLK": "Technology", "ARKK": "Innovation", "GDX": "Gold Miners",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "AVIX": "Volatility Index",
    "DJT": "Media", "WOLF": "Semiconductors"
}

# Date range
days_back = 10
end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

# Download and clean data
print(f"\nFetching data from {start_date.date()} to {end_date.date()}...\n")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date)

# Use Adj Close if available
if isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
elif 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    print("\n[!] Warning: 'Adj Close' not found. Using 'Close' prices instead.\n")
    data = data['Close']

# Calculate metrics
returns = data.pct_change().dropna()
weekly_returns = returns.sum() * 100
volatility = returns.std() * 100

# Detect rising trend for DJT and WOLF
realtime_trends = {}
for stock in ["DJT", "WOLF"]:
    if stock in data.columns:
        last_two = data[stock].dropna()[-2:]
        if len(last_two) == 2:
            trend = "RISING" if last_two.iloc[-1] > last_two.iloc[-2] else "FALLING"
        else:
            trend = "UNKNOWN"
        realtime_trends[stock] = trend

# Build dataframe
df = pd.DataFrame({
    "Asset Type": [tickers[t] for t in tickers],
    "Weekly Return (%)": weekly_returns.round(2),
    "Volatility (%)": volatility.round(2),
    "Trend": [realtime_trends.get(t, "") for t in tickers]
}, index=tickers.keys())

# Add signal logic
def signal_logic(row):
    if row["Weekly Return (%)"] > 5 and row["Volatility (%)"] < 4:
        return "BUY"
    elif row["Weekly Return (%)"] > 0:
        return "NEUTRAL"
    else:
        return "SELL"

df["Signal"] = df.apply(signal_logic, axis=1)
df = df.sort_values(by="Weekly Return (%)", ascending=False)

# Save to CSV
df.to_csv("alladin_dashboard_v2.csv")
print("\nDashboard saved as 'alladin_dashboard_v2.csv'\n")

# Color-coded terminal output
for _, row in df.iterrows():
    line = f"{row.name}: {row['Signal']} | Return: {row['Weekly Return (%)']}% | Volatility: {row['Volatility (%)']}% | Trend: {row['Trend']}"
    if row["Signal"] == "BUY":
        print(Fore.GREEN + line)
    elif row["Signal"] == "SELL":
        print(Fore.RED + line)
    elif row["Signal"] == "NEUTRAL":
        print(Fore.YELLOW + line)
    else:
        print(line)

