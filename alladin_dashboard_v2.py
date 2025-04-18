#!/usr/bin/env python3
import os
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import telegram
from sklearn.decomposition import PCA
import umap

# === Configuration ===
TICKERS = ["TSLA", "AAPL", "MSFT", "DJT", "WOLF", "CL=F", "NG=F"]
OPTION_TICKERS = ["TSLA", "CL=F", "NG=F"]

PERIOD = "5d"
INTERVAL = "5m"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None

# === Data & Indicator Pipeline ===
def fetch_data(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    df.dropna(inplace=True)
    return df

def compute_indicators(df):
    df["return_1"] = df["Close"].pct_change()
    # RSI
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    df["RSI"] = 100 - (100 / (1 + roll_up/roll_down))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_high"] = sma20 + 2*std20
    df["BB_low"]  = sma20 - 2*std20

    df.dropna(inplace=True)
    return df

def build_feature_matrix(tickers):
    rows, idx = [], []
    for t in tickers:
        df = fetch_data(t)
        df = compute_indicators(df)
        last = df.iloc[-1]
        feats = [
            last["return_1"],
            last["RSI"],
            last["MACD"] - last["MACD_signal"],    # MACD histogram
            (last["Close"] - last["BB_low"]) /     # %B
                (last["BB_high"] - last["BB_low"]),
            df["Close"].rolling(50).std().iloc[-1], # 50-period vol
        ]
        rows.append(feats); idx.append(t)
    return pd.DataFrame(rows, index=idx,
        columns=["ret1","RSI","MACD_hist","BB_pct","vol50"])

# === PCA + UMAP ===
def apply_pca_umap(features):
    pca = PCA(n_components=3)
    pc = pca.fit_transform(features)
    pca_df = pd.DataFrame(pc, index=features.index,
                          columns=["PC1","PC2","PC3"])

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(pca_df)
    umap_df = pd.DataFrame(umap_emb, index=features.index,
                           columns=["UMAP1","UMAP2"])

    return features.join(pca_df).join(umap_df), pca, reducer

# === Signal Logic ===
def generate_signals(feature_matrix):
    sig = {}
    for t, row in feature_matrix.iterrows():
        score = row["PC1"] + 0.5 * row["PC2"]
        if score > 1.0:
            sig[t] = "BUY"
        elif score < -1.0:
            sig[t] = "SELL"
        else:
            sig[t] = "HOLD"
    return sig

def generate_option_notifications(tickers):
    alerts = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            exps = tk.options
            if not exps:
                continue
            exp = exps[0]  # nearest expiry
            chain = tk.option_chain(exp)
            call_vol = chain.calls["volume"].sum()
            put_vol  = chain.puts["volume"].sum()
            # avoid div-by-zero
            if put_vol == 0 and call_vol == 0:
                continue
            ratio = call_vol / put_vol if put_vol else np.inf

            if ratio > 1.2:
                alerts[t] = f"CALL ALERT ({ratio:.2f}× calls vs puts)"
            elif ratio < 0.8:
                alerts[t] = f"PUT ALERT  ({ratio:.2f}× calls vs puts)"
        except Exception as e:
            # yfinance may not support futures options (CL=F, NG=F)
            continue
    return alerts

# === Alerts ===
def send_alerts(signals, option_alerts):
    lines = []
    for t, s in signals.items():
        line = f"{t}: {s}"
        if t in option_alerts:
            line += f" | {option_alerts[t]}"
        lines.append(line)

    if bot:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(lines))
    else:
        print("\n".join(lines))

# === Main ===
def main():
    feats = build_feature_matrix(TICKERS)
    feats_exp, pca_model, umap_model = apply_pca_umap(feats)
    print("Expanded features:\n", feats_exp)

    signals = generate_signals(feats_exp)
    option_alerts = generate_option_notifications(OPTION_TICKERS)
    send_alerts(signals, option_alerts)

if __name__ == "__main__":
    main()
