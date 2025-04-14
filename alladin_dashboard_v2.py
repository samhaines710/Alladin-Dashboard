import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import pytz

# === Telegram Setup ===
try:
    import telegram
    TELEGRAM_ENABLED = True
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None
except ImportError:
    print("Telegram module not found. Alerts disabled.")
    TELEGRAM_ENABLED = False
    bot = None

# === Ticker Categories ===
COMMODITIES = ['CL=F', 'NG=F', 'ETH-USD', 'BTC-USD']
STOCKS = ['DJT', 'WOLF', 'LMT', 'AAPL', 'TSLA', 'DOT']
ETFS = ['SPY', 'IVV']  # Add more ETFs as needed
TICKERS = COMMODITIES + STOCKS + ETFS
# ALWAYS_ON: These run 24/7 (commodities and ETFs)
ALWAYS_ON = COMMODITIES + ETFS

# === News/Sentiment Intelligence (Placeholder for Stocks) ===
def get_news_sentiment(ticker):
    # Placeholder: Replace with real news/sentiment API or scraping.
    preset = {
        'DJT': 'Bearish',   # DJT has negative news currently.
        'WOLF': 'Neutral',
        'LMT': 'Neutral',
        'AAPL': 'Bullish',
        'TSLA': 'Neutral',
        'DOT': 'Neutral',
        'SPY': 'Neutral',
        'IVV': 'Neutral'
    }
    return preset.get(ticker, "Neutral")

# === RSA Market Hours Logic (SAST: 15:30–22:00 for stocks/ETFs) ===
def market_is_open(ticker):
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    rsa = pytz.timezone("Africa/Johannesburg")
    now = utc_now.astimezone(rsa)
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    if weekday >= 5:
        return False
    # Stocks and ETFs only trade during these hours.
    if ticker in STOCKS or ticker in ETFS:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)
    # Commodities trade 24/7.
    return True

# === Indicators Functions ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
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

# === Data Fetching ===
def fetch_data(ticker, interval='15m', period='2d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is None or df.empty or len(df) < 30:
            print(f"[{ticker}] No data available. Skipping.")
            return None
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])
        return df.dropna()
    except Exception as e:
        print(f"[{ticker}] Error fetching data: {e}")
        return None

# === Options Evaluation for Commodities (NG and Oil) ===
def evaluate_options(ticker, df, signal_type):
    # Only apply to commodities (oil and natural gas)
    if ticker not in ['CL=F', 'NG=F']:
        return None
    current_price = float(df.iloc[-1]['Close'])
    if signal_type in ["STRONG BUY", "BUY"]:
        entry = current_price
        target = current_price * 1.02  # example: 2% profit target
        stop = current_price * 0.99    # example: 1% stop loss
        return f"{ticker}: CALL OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    elif signal_type in ["STRONG SELL", "SELL"]:
        entry = current_price
        target = current_price * 0.98  # example: 2% profit target for put
        stop = current_price * 1.01    # example: 1% stop loss
        return f"{ticker}: PUT OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    return None

# === Signal & Trend Reversal Evaluation ===
def evaluate_signals(df, ticker):
    if df is None or len(df) < 5:
        return None

    try:
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        prev2 = df.iloc[-3]
        prev3 = df.iloc[-4]

        close_now = float(latest['Close'])
        close_prev = float(previous['Close'])
        price_change = ((close_now - close_prev) / close_prev) * 100

        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        signal = float(latest['Signal'])
    except Exception as e:
        print(f"[{ticker}] Error extracting float values: {e}")
        return None

    signal_type = None
    reasons = []

    # --- Adaptive Thresholds: Commodities/ETFs vs Stocks ---
    if ticker in COMMODITIES or ticker in ETFS:
        if price_change > 1.0 and rsi > 55 and macd > signal:
            signal_type = "STRONG BUY"
            reasons.append(f"RSI {rsi:.1f}")
            reasons.append("MACD crossover")
            reasons.append(f"+{price_change:.2f}%")
        elif price_change < -1.5 and rsi < 45 and macd < signal:
            signal_type = "STRONG SELL"
            reasons.append(f"RSI {rsi:.1f}")
            reasons.append("MACD downward")
            reasons.append(f"{price_change:.2f}%")
        elif macd > signal and price_change > 0.5:
            signal_type = "BUY"
            reasons.append("MACD rising")
            reasons.append(f"+{price_change:.2f}%")
        elif macd < signal and price_change < -0.5:
            signal_type = "SELL"
            reasons.append("MACD falling")
            reasons.append(f"{price_change:.2f}%")
    elif ticker in STOCKS:
        if price_change > 1.5 and rsi > 60 and macd > signal:
            signal_type = "STRONG BUY"
            reasons.append(f"RSI {rsi:.1f}")
            reasons.append("MACD crossover")
            reasons.append(f"+{price_change:.2f}%")
        elif price_change < -2.0 and rsi < 40 and macd < signal:
            signal_type = "STRONG SELL"
            reasons.append(f"RSI {rsi:.1f}")
            reasons.append("MACD downward")
            reasons.append(f"{price_change:.2f}%")
        elif macd > signal and price_change > 1.0:
            signal_type = "BUY"
            reasons.append("MACD rising")
            reasons.append(f"+{price_change:.2f}%")
        elif macd < signal and price_change < -1.0:
            signal_type = "SELL"
            reasons.append("MACD falling")
            reasons.append(f"{price_change:.2f}%")
        # Incorporate news/sentiment intelligence for stocks.
        sentiment = get_news_sentiment(ticker)
        if sentiment != "Neutral":
            reasons.append(f"News: {sentiment}")
            if sentiment == "Bearish" and signal_type in ["STRONG BUY", "BUY"]:
                signal_type = "SUSPEND BUY"
                reasons.append("(Negative news override)")
            elif sentiment == "Bullish" and signal_type in ["STRONG SELL", "SELL"]:
                signal_type = "SUSPEND SELL"
                reasons.append("(Positive news override)")

    # --- Trend Reversal Detection (applies to all tickers) ---
    try:
        if prev3['MACD'].item() < prev3['Signal'].item() and \
           prev2['MACD'].item() < prev2['Signal'].item() and \
           macd > signal and rsi > prev2['RSI'].item():
            reversal_msg = f"{ticker}: REVERSAL UP | RSI & MACD rising | WATCH @ {close_now:.2f}"
            send_telegram_alert(reversal_msg)
        elif prev3['MACD'].item() > prev3['Signal'].item() and \
             prev2['MACD'].item() > prev2['Signal'].item() and \
             macd < signal and rsi < prev2['RSI'].item():
            reversal_msg = f"{ticker}: REVERSAL DOWN | RSI & MACD falling | WATCH @ {close_now:.2f}"
            send_telegram_alert(reversal_msg)
    except Exception as e:
        print(f"[{ticker}] Reversal check failed: {e}")

    # If there is a signal for commodities/ETFs, add options evaluation.
    options_signal = None
    if ticker in COMMODITIES:
        options_signal = evaluate_options(ticker, df, signal_type)
        if options_signal:
            send_telegram_alert(options_signal)

    if signal_type:
        message = f"{ticker}: {signal_type} | {' | '.join(reasons)}"
        send_telegram_alert(message)
        return message

    return None

# === Telegram Alert Function ===
def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Options Evaluation for Commodities ===
def evaluate_options(ticker, df, signal_type):
    # Only for oil and natural gas.
    if ticker not in ['CL=F', 'NG=F']:
        return None
    current_price = float(df.iloc[-1]['Close'])
    if signal_type in ["STRONG BUY", "BUY"]:
        entry = current_price
        target = current_price * 1.02  # e.g., 2% above as a profit target
        stop = current_price * 0.99    # e.g., 1% below as stop loss
        return f"{ticker}: CALL OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    elif signal_type in ["STRONG SELL", "SELL"]:
        entry = current_price
        target = current_price * 0.98  # e.g., 2% lower for put option target
        stop = current_price * 1.01    # e.g., 1% above as stop loss
        return f"{ticker}: PUT OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    return None

# === Main Execution ===
def main():
    rsa_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    print(f"Running Alladin Dashboard at RSA time: {rsa_now.strftime('%Y-%m-%d %H:%M:%S')}")
    alerts = []
    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market closed. Skipping.")
            continue
        df = fetch_data(ticker)
        result = evaluate_signals(df, ticker)
        if result:
            alerts.append(result)
    if alerts:
        print("\n--- Alerts ---")
        for alert in alerts:
            print(alert)
    else:
        print("No signals triggered.")

if __name__ == "__main__":
    main()

# === News/Sentiment Intelligence Placeholder ===
def get_news_sentiment(ticker):
    # Placeholder: Replace with real news/sentiment API logic.
    preset = {
        'DJT': 'Bearish',
        'WOLF': 'Neutral',
        'LMT': 'Neutral',
        'AAPL': 'Bullish',
        'TSLA': 'Neutral',
        'DOT': 'Neutral',
        'SPY': 'Neutral',
        'IVV': 'Neutral'
    }
    return preset.get(ticker, "Neutral")
