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
ETFS = ['SPY', 'IVV']
TICKERS = COMMODITIES + STOCKS + ETFS
ALWAYS_ON = COMMODITIES + ETFS  # Runs 24/7 (commodities, crypto, ETFs)

# === News/Sentiment Intelligence (Placeholder) ===
def get_news_sentiment(ticker):
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

# === RSA Market Hours ===
def market_is_open(ticker):
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    rsa = pytz.timezone("Africa/Johannesburg")
    now = utc_now.astimezone(rsa)
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    if weekday >= 5:
        return False
    # Stocks & ETFs: 15:30–22:00 SAST
    if ticker in STOCKS or ticker in ETFS:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)
    # Commodities & crypto: 24/7
    return True

# === Indicator Functions ===
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

def compute_atr(df, period=14):
    """
    Computes ATR without .combine(). No truth-value errors.
    """
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['range1'] = df['High'] - df['Low']
    df['range2'] = (df['High'] - df['prev_close']).abs()
    df['range3'] = (df['Low'] - df['prev_close']).abs()
    df['tr'] = df[['range1','range2','range3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']

# === Data Fetching ===
def fetch_data(ticker, interval='15m', period='2d'):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is None or df.empty or len(df) < 30:
            print(f"[{ticker}] No data available. Skipping.")
            return None

        # Basic changes
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])

        # SAFE assignment of MACD & Signal
        macd_series, signal_series = compute_macd(df['Close'])
        df['MACD'] = macd_series
        df['Signal'] = signal_series

        return df.dropna()
    except Exception as e:
        print(f"[{ticker}] Error fetching data: {e}")
        return None

# === Options for Commodities (NG, Oil) ===
def evaluate_options(ticker, df, signal_type):
    if ticker not in ['CL=F','NG=F']:
        return None
    current_price = float(df.iloc[-1]['Close'])
    atr_series = compute_atr(df)
    if atr_series.isna().all():
        return None

    atr_val = atr_series.iloc[-1]
    # Base thresholds for calm markets
    target_pct = 0.02
    stop_pct = 0.01
    # Expand thresholds if ATR > 1% of price
    if atr_val / current_price > 0.01:
        target_pct = 0.03
        stop_pct = 0.015

    if signal_type in ["STRONG BUY","BUY"]:
        entry = current_price
        target = current_price*(1 + target_pct)
        stop = current_price*(1 - stop_pct)
        return f"{ticker}: CALL OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    elif signal_type in ["STRONG SELL","SELL"]:
        entry = current_price
        target = current_price*(1 - target_pct)
        stop = current_price*(1 + stop_pct)
        return f"{ticker}: PUT OPTION | Entry: {entry:.2f} | Target: {target:.2f} | Stop: {stop:.2f}"
    return None

# === Signal & Reversal Evaluation ===
def evaluate_signals(df, ticker):
    if df is None or len(df) < 5:
        return None

    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        prev3 = df.iloc[-4]

        close_now = float(latest['Close'])
        close_prev = float(prev['Close'])
        price_change = ((close_now - close_prev)/close_prev)*100

        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        signal = float(latest['Signal'])
    except Exception as e:
        print(f"[{ticker}] Error extracting float values: {e}")
        return None

    signal_type = None
    reasons = []

    # Commodities & ETFs have more sensitive triggers
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

    # Stocks have looser thresholds
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

        # News override
        sentiment = get_news_sentiment(ticker)
        if sentiment != "Neutral":
            reasons.append(f"News: {sentiment}")
            if sentiment == "Bearish" and signal_type in ["STRONG BUY","BUY"]:
                signal_type = "SUSPEND BUY"
                reasons.append("(Negative news override)")
            elif sentiment == "Bullish" and signal_type in ["STRONG SELL","SELL"]:
                signal_type = "SUSPEND SELL"
                reasons.append("(Positive news override)")

    # Reversal detection
    try:
        if prev3['MACD'] < prev3['Signal'] and prev2['MACD'] < prev2['Signal'] and macd > signal and rsi > prev2['RSI']:
            rev_msg = f"{ticker}: REVERSAL UP | RSI & MACD rising | WATCH @ {close_now:.2f}"
            send_telegram_alert(rev_msg)
        elif prev3['MACD'] > prev3['Signal'] and prev2['MACD'] > prev2['Signal'] and macd < signal and rsi < prev2['RSI']:
            rev_msg = f"{ticker}: REVERSAL DOWN | RSI & MACD falling | WATCH @ {close_now:.2f}"
            send_telegram_alert(rev_msg)
    except Exception as e:
        print(f"[{ticker}] Reversal check failed: {e}")

    # Commodity options signals
    if ticker in COMMODITIES:
        opt_signal = evaluate_options(ticker, df, signal_type)
        if opt_signal:
            send_telegram_alert(opt_signal)

    # If final signal
    if signal_type:
        msg = f"{ticker}: {signal_type} | {' | '.join(reasons)}"
        send_telegram_alert(msg)
        return msg

    return None

# === Telegram Alerts ===
def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Main ===
def main():
    local_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    print(f"Running Alladin Dashboard at RSA time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
    alerts = []

    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market is closed. Skipping.")
            continue

        df = fetch_data(ticker)
        result = evaluate_signals(df, ticker)
        if result:
            alerts.append(result)

    if alerts:
        print("\n--- Alerts ---")
        for a in alerts:
            print(a)
    else:
        print("No signals triggered.")

if __name__ == "__main__":
    main()
