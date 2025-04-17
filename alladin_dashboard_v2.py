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
ALWAYS_ON = COMMODITIES + ETFS

peak_tracker = {}

# === Market Hours (RSA) ===
def market_is_open(ticker):
    utc_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    rsa = pytz.timezone("Africa/Johannesburg")
    now = utc_now.astimezone(rsa)
    wd = now.weekday()
    hour = now.hour
    minute = now.minute
    if wd >= 5:
        return False
    if ticker in STOCKS or ticker in ETFS:
        return (hour == 15 and minute >= 30) or (16 <= hour < 22)
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
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['range1'] = df['High'] - df['Low']
    df['range2'] = (df['High'] - df['prev_close']).abs()
    df['range3'] = (df['Low'] - df['prev_close']).abs()
    df['tr'] = df[['range1','range2','range3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']

# === Telegram Alerts ===
def send_telegram_alert(message):
    if TELEGRAM_ENABLED and bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram error: {e}")

# === Main Execution ===
def main():
    local_now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(pytz.timezone("Africa/Johannesburg"))
    print(f"Running Alladin at RSA time: {local_now.strftime('%Y-%m-%d %H:%M:%S')}")
    for ticker in TICKERS:
        if not market_is_open(ticker):
            print(f"[{ticker}] Market closed. Skipping.")
            continue

        try:
            df = yf.download(ticker, interval='5m', period='1d', progress=False)
            if df is None or df.empty or len(df) < 20:
                print(f"[{ticker}] No data or insufficient rows.")
                continue

            df['RSI'] = compute_rsi(df['Close'])
            macd, signal = compute_macd(df['Close'])
            df['MACD'] = macd
            df['Signal'] = signal
            df.dropna(inplace=True)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            rsi_now = last['RSI']
            macd_now = last['MACD']
            sig_now = last['Signal']
            price_now = last['Close']
            price_prev = prev['Close']

            if ticker not in peak_tracker:
                peak_tracker[ticker] = {'price': 0, 'rsi': 0, 'time': None, 'macd': 0, 'signal': 0, 'status': 'watching', 'fallback_count': 0}

            tracker = peak_tracker[ticker]

            # Peak detection
            if price_now > tracker['price'] and rsi_now > 65 and macd_now > sig_now:
                peak_tracker[ticker] = {'price': price_now, 'rsi': rsi_now, 'time': local_now, 'macd': macd_now, 'signal': sig_now, 'status': 'peaked', 'fallback_count': 0}
                send_telegram_alert(f"{ticker}: LOCAL PEAK DETECTED @ {price_now:.2f} | RSI {rsi_now:.1f} | MACD {macd_now:.2f}")

            elif tracker['status'] == 'peaked' and price_now < tracker['price'] * 0.995 and rsi_now < tracker['rsi'] - 3:
                send_telegram_alert(f"{ticker}: DIP FROM PEAK @ {price_now:.2f} | Watching for Rebound | Previous Peak: {tracker['price']:.2f}")
                peak_tracker[ticker]['status'] = 'dipped'
                peak_tracker[ticker]['fallback_count'] = 0

            elif tracker['status'] == 'dipped':
                if macd_now > sig_now and rsi_now > prev['RSI']:
                    send_telegram_alert(f"{ticker}: EARLY CALL ZONE | Re-Peak Expected | Entry @ {price_now:.2f} | Previous Peak {tracker['price']:.2f}")
                    peak_tracker[ticker]['status'] = 'watching'
                else:
                    peak_tracker[ticker]['fallback_count'] += 1
                    if peak_tracker[ticker]['fallback_count'] >= 3:
                        send_telegram_alert(f"{ticker}: ABORT CALL ZONE | Momentum Lost After Dip | Avoid Entry @ {price_now:.2f}")
                        peak_tracker[ticker]['status'] = 'watching'

        except Exception as e:
            print(f"[{ticker}] Data error: {e}")

if __name__ == "__main__":
    main()
