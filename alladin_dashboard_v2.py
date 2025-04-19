#!/usr/bin/env python3
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import telegram
from sklearn.decomposition import PCA
import umap

# === Configuration ===
TICKERS = ["TSLA", "AAPL", "MSFT", "DJT", "WOLF"]
OPTION_TICKERS = ["TSLA"]
PERIOD = "5d"
INTERVAL = "5m"

# === Telegram Bot Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None

# === Polygon API ===
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# === Fetch Candles via Polygon ===
def fetch_data(ticker):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/5/day?adjusted=true&sort=desc&limit=100&apiKey={POLYGON_API_KEY}"
    try:
        res = requests.get(url)
        data = res.json()
        if "results" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)
        df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]]
    except:
        return pd.DataFrame()

# === Indicator Calculation ===
def compute_indicators(df):
    df["return_1"] = df["Close"].pct_change()
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    df["RSI"] = 100 - (100 / (1 + roll_up/roll_down))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_high"] = sma20 + 2*std20
    df["BB_low"] = sma20 - 2*std20
    return df.dropna()

# === Feature Matrix ===
def build_feature_matrix(tickers):
    rows, idx = [], []
    for t in tickers:
        df = fetch_data(t)
        if df.empty: continue
        df = compute_indicators(df)
        last = df.iloc[-1]
        feats = [
            last["return_1"],
            last["RSI"],
            last["MACD"] - last["MACD_signal"],
            (last["Close"] - last["BB_low"]) / (last["BB_high"] - last["BB_low"]),
            df["Close"].rolling(50).std().iloc[-1],
        ]
        rows.append(feats); idx.append(t)
    return pd.DataFrame(rows, index=idx, columns=["ret1","RSI","MACD_hist","BB_pct","vol50"])

# === PCA + UMAP Logic ===
def apply_pca_umap(features):
    if features.shape[0] < 2:
        print(f"â ï¸ Not enough data for PCA/UMAP. Rows: {features.shape[0]}")
        return features, None, None
    try:
        X = features.to_numpy(dtype=float)
        pca = PCA(n_components=min(3, X.shape[0]))
        pc = pca.fit_transform(X)
        pca_df = pd.DataFrame(pc, index=features.index, columns=[f"PC{i+1}" for i in range(pc.shape[1])])
        um = umap.UMAP(n_components=2, random_state=42)
        emb = um.fit_transform(pc)
        umap_df = pd.DataFrame(emb, index=features.index, columns=["UMAP1","UMAP2"])
        return features.join(pca_df).join(umap_df), pca, um
    except Exception as e:
        print(f"â PCA/UMAP failed: {e}")
        return features, None, None

# === Signal Logic ===
def generate_signals(df):
    sig = {}
    for t, row in df.iterrows():
        score = row.get("PC1", 0) + 0.5 * row.get("PC2", 0)
        sig[t] = "BUY" if score > 1 else "SELL" if score < -1 else "HOLD"
    return sig

# === TSLA Deep Option Flow ===
def tsla_option_flow():
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/TSLA?limit=1000&apiKey={POLYGON_API_KEY}"
        r = requests.get(url)
        data = r.json()
        if "results" not in data: return ""

        df = pd.DataFrame(data["results"])
        df["type"] = df["details"].apply(lambda x: x["contract_type"])
        df["strike"] = df["details"].apply(lambda x: x["strike_price"])
        df["vol"] = df["day"].apply(lambda x: x["volume"])
        df["iv"] = df["greeks"].apply(lambda x: x.get("implied_volatility", None))
        df["delta"] = df["greeks"].apply(lambda x: x.get("delta", None))

        calls = df[df["type"]=="call"].sort_values(by="vol", ascending=False).head(3)
        puts  = df[df["type"]=="put"].sort_values(by="vol", ascending=False).head(3)

        msg = "
Top TSLA Calls:
" + "
".join([f"â¢ {int(row.strike)}C | Vol: {int(row.vol)} | IV: {row.iv:.0%} | Î: {row.delta:.2f}" for _, row in calls.iterrows()])
        msg += "
Top TSLA Puts:
" + "
".join([f"â¢ {int(row.strike)}P | Vol: {int(row.vol)} | IV: {row.iv:.0%} | Î: {row.delta:.2f}" for _, row in puts.iterrows()])
        return msg
    except:
        return ""

# === Alert Sending ===
def send_alerts(signals):
    if not signals: return
    lines = [f"{t}: {s}" for t, s in signals.items()]
    lines.append(tsla_option_flow())
    msg = "
".join(lines)
    if bot and msg.strip():
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            print(f"Telegram error: {e}")
    else:
        print(msg)

# === Main ===
def main():
    feats = build_feature_matrix(TICKERS)
    feats_exp, pca_model, umap_model = apply_pca_umap(feats)
    signals = generate_signals(feats_exp)
    send_alerts(signals)

if __name__ == "__main__":
    main()
