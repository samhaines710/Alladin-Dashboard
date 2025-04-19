#!/usr/bin/env python3
import os
import requests
import pandas as pd
import numpy as np
import telegram
from sklearn.decomposition import PCA
import umap

# === Configuration ===
TICKERS = ["TSLA", "AAPL", "MSFT", "DJT", "WOLF"]
OPTION_TICKERS = ["TSLA", "AAPL", "WOLF"]
PERIOD = "5d"
INTERVAL = "5m"

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None

# === Polygon API Key ===
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# === Fetch 5m Candles via Polygon ===
def fetch_data(ticker):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/"
        f"range/5/minute/5/day"
        f"?adjusted=true&sort=desc&limit=100&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url)
        data = res.json()
        if "results" not in data:
            print(f"⚠️ No data for {ticker}")
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("t", inplace=True)
        df.rename(
            columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"},
            inplace=True
        )
        return df[["Open","High","Low","Close","Volume"]]
    except Exception as e:
        print(f"❌ fetch_data error for {ticker}: {e}")
        return pd.DataFrame()

# === Compute Standard Indicators ===
def compute_indicators(df):
    df = df.copy()
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

# === Build Feature Matrix ===
def build_feature_matrix(tickers):
    rows, idx = [], []
    for t in tickers:
        df = fetch_data(t)
        if df.empty:
            continue
        df = compute_indicators(df)
        last = df.iloc[-1]
        feats = [
            last["return_1"],
            last["RSI"],
            last["MACD"] - last["MACD_signal"],
            (last["Close"] - last["BB_low"]) / (last["BB_high"] - last["BB_low"]),
            df["Close"].rolling(50).std().iloc[-1]
        ]
        rows.append(feats)
        idx.append(t)
    return pd.DataFrame(rows, index=idx,
                        columns=["ret1","RSI","MACD_hist","BB_pct","vol50"])

# === Failsafe PCA + UMAP ===
def apply_pca_umap(features):
    if features.shape[0] < 2:
        print(f"⚠️ Not enough rows for PCA/UMAP: {features.shape[0]}")
        return features, None, None
    try:
        X = features.to_numpy(dtype=float)
        comps = min(3, X.shape[0])
        pca = PCA(n_components=comps)
        pc = pca.fit_transform(X)
        pca_df = pd.DataFrame(pc,
                              index=features.index,
                              columns=[f"PC{i+1}" for i in range(pc.shape[1])])
        um = umap.UMAP(n_components=2, random_state=42)
        emb = um.fit_transform(pc)
        umap_df = pd.DataFrame(emb,
                               index=features.index,
                               columns=["UMAP1","UMAP2"])
        return features.join(pca_df).join(umap_df), pca, um
    except Exception as e:
        print(f"PCA/UMAP error: {e}")
        return features, None, None

# === Generate BUY/HOLD/SELL Signals ===
def generate_signals(df):
    sig = {}
    for t, row in df.iterrows():
        score = row.get("PC1", 0) + 0.5 * row.get("PC2", 0)
        sig[t] = "BUY" if score > 1 else "SELL" if score < -1 else "HOLD"
    return sig

# === Deep Option‑Flow Summary ===
def option_flow_summary(ticker):
    try:
        url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}"
            f"?limit=1000&apiKey={POLYGON_API_KEY}"
        )
        r = requests.get(url)
        data = r.json()
        if "results" not in data:
            return ""
        df = pd.DataFrame(data["results"])
        df["type"]   = df["details"].apply(lambda x: x["contract_type"])
        df["strike"] = df["details"].apply(lambda x: x["strike_price"])
        df["vol"]    = df["day"].apply(lambda x: x["volume"])
        df["iv"]     = df["greeks"].apply(lambda x: x.get("implied_volatility"))
        df["delta"]  = df["greeks"].apply(lambda x: x.get("delta"))
        calls = df[df["type"]=="call"].nlargest(3, "vol")
        puts  = df[df["type"]=="put"].nlargest(3, "vol")
        msg  = f"\\nTop {ticker} Calls:\\n" + "\\n".join(
            f"• {int(r.strike)}C | Vol: {int(r.vol)} | IV: {r.iv:.0%} | Δ: {r.delta:.2f}"
            for _, r in calls.iterrows()
        )
        msg += f"\\nTop {ticker} Puts:\\n" + "\\n".join(
            f"• {int(r.strike)}P | Vol: {int(r.vol)} | IV: {r.iv:.0%} | Δ: {r.delta:.2f}"
            for _, r in puts.iterrows()
        )
        return msg
    except Exception as e:
        print(f"Option flow error for {ticker}: {e}")
        return ""

# === Send Alerts with Guard ===
def send_alerts(signals):
    lines = [f"{t}: {s}" for t, s in signals.items()]
    for tk in OPTION_TICKERS:
        summary = option_flow_summary(tk)
        if summary:
            lines.append(summary)
    msg = "\n".join(lines).strip()
    if bot and msg:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            print(f"Telegram error: {e}")
    else:
        print("No message sent (empty or bot not configured).")

# === Main ===
def main():
    feats, = (None,)  # dummy to avoid lint error if cron fails early
    feats = build_feature_matrix(TICKERS)
    feats_exp, pca_model, umap_model = apply_pca_umap(feats)
    signals = generate_signals(feats_exp)
    send_alerts(signals)

if __name__ == "__main__":
    main()
