#!/usr/bin/env python3
import os
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
    return df.dropna()

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
    df["BB_high"] = sma20 + 2 * std20
    df["BB_low"] = sma20 - 2 * std20
    return df.dropna()

def build_feature_matrix(tickers):
    rows, idx = [], []
    for t in tickers:
        df = fetch_data(t)
        df = compute_indicators(df)
        if df.empty:
            continue
        last = df.iloc[-1]
        feats = [
            last["return_1"],
            last["RSI"],
            last["MACD"] - last["MACD_signal"],
            (last["Close"] - last["BB_low"]) / (last["BB_high"] - last["BB_low"]),
            df["Close"].rolling(50).std().iloc[-1],
        ]
        rows.append(feats)
        idx.append(t)
    return pd.DataFrame(rows, index=idx,
                        columns=["ret1", "RSI", "MACD_hist", "BB_pct", "vol50"])

# === PCA + UMAP (Failsafe) ===
def apply_pca_umap(features):
    try:
        if features.empty or features.shape[0] < 2:
            print(f"⚠️ Not enough data for PCA/UMAP. Rows: {features.shape[0]}")
            return features, None, None

        try:
            X = features.to_numpy(dtype=float)
        except Exception as e:
            print(f"⚠️ Coercing features failed: {e}")
            features = features.apply(pd.to_numeric, errors="coerce").dropna(how="any")
            if features.empty or features.shape[0] < 2:
                print(f"⚠️ Data too small even after coercion. Rows: {features.shape[0]}")
                return features, None, None
            X = features.to_numpy(dtype=float)

        comps = min(3, X.shape[0])
        pca = PCA(n_components=comps)
        try:
            pc = pca.fit_transform(X)
        except Exception as e:
            print(f"⚠️ PCA failed: {e}")
            return features, None, None

        pca_df = pd.DataFrame(pc, index=features.index,
                              columns=[f"PC{i+1}" for i in range(comps)])

        try:
            um = umap.UMAP(n_components=2, random_state=42)
            emb = um.fit_transform(pc)
            umap_df = pd.DataFrame(emb, index=features.index,
                                   columns=["UMAP1", "UMAP2"])
        except Exception as e:
            print(f"⚠️ UMAP failed: {e}")
            return features.join(pca_df), pca, None

        return features.join(pca_df).join(umap_df), pca, um

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR in apply_pca_umap: {e}")
        return features, None, None

# === Signal Logic ===
def generate_signals(feature_matrix):
    sig = {}
    has_pca = any(col.startswith("PC") for col in feature_matrix.columns)
    for t, row in feature_matrix.iterrows():
        if not has_pca:
            sig[t] = "HOLD"
            continue
        score = row.get("PC1", 0) + 0.5 * row.get("PC2", 0)
        sig[t] = "BUY" if score > 1 else "SELL" if score < -1 else "HOLD"
    return sig

# === Option Flow Alerts ===
def generate_option_notifications(tickers):
    alerts = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            exps = tk.options
            if not exps:
                continue
            chain = tk.option_chain(exps[0])
            call_vol = chain.calls["volume"].sum()
            put_vol = chain.puts["volume"].sum()
            if call_vol + put_vol == 0:
                continue
            ratio = call_vol / put_vol if put_vol else np.inf
            if ratio > 1.2:
                alerts[t] = f"CALL ALERT ({ratio:.2f}×)"
            elif ratio < 0.8:
                alerts[t] = f"PUT ALERT  ({ratio:.2f}×)"
        except Exception:
            pass
    return alerts

# === Alert Sending ===
def send_alerts(signals, opt_alerts):
    lines = []
    for t, s in signals.items():
        line = f"{t}: {s}"
        if t in opt_alerts:
            line += f" | {opt_alerts[t]}"
        lines.append(line)
    msg = "\n".join(lines)
    if bot:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    else:
        print(msg)

# === Main Execution ===
def main():
    feats = build_feature_matrix(TICKERS)
    feats_exp, pca_model, umap_model = apply_pca_umap(feats)
    print("Expanded features:\n", feats_exp)
    signals = generate_signals(feats_exp)
    option_alerts = generate_option_notifications(OPTION_TICKERS)
    send_alerts(signals, option_alerts)

if __name__ == "__main__":
    main()
