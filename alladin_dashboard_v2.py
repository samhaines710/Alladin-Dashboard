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

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None

# === Data & Indicator Pipeline ===
def fetch_data(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    return df.dropna()

def compute_indicators(df):
    df = df.copy()
    df["return_1"] = df["Close"].pct_change()
    # RSI
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up   = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    df["RSI"] = 100 - (100 / (1 + roll_up/roll_down))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    # Bollinger Bands
    sma20     = df["Close"].rolling(20).mean()
    std20     = df["Close"].rolling(20).std()
    df["BB_high"] = sma20 + 2*std20
    df["BB_low"]  = sma20 - 2*std20
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
        rows.append(feats); idx.append(t)
    return pd.DataFrame(rows, index=idx,
        columns=["ret1","RSI","MACD_hist","BB_pct","vol50"])

# === PCA + UMAP (patched) ===
def apply_pca_umap(features):
    # 1) If we have no rows, just return features unmodified
    if features.shape[0] == 0:
        return features, None, None

    # 2) Force a clean float64 array
    try:
        X = features.to_numpy(dtype=float)
    except Exception as e:
        print("⚠️ Coercing to numeric:", e)
        features = features.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        X = features.to_numpy(dtype=float)

    # 3) PCA
    pca = PCA(n_components=3)
    pc  = pca.fit_transform(X)
    pca_df = pd.DataFrame(pc, index=features.index,
                          columns=["PC1","PC2","PC3"])

    # 4) UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(pc)
    umap_df = pd.DataFrame(emb, index=features.index,
                           columns=["UMAP1","UMAP2"])

    # 5) Merge back
    return features.join(pca_df).join(umap_df), pca, reducer

# === Signals & Alerts ===
def generate_signals(feat):
    sig = {}
    for t, row in feat.iterrows():
        # If PCA was unavailable, default to HOLD
        if "PC1" not in row:
            sig[t] = "HOLD"
            continue
        score = row["PC1"] + 0.5*row["PC2"]
        sig[t] = "BUY" if score>1 else "SELL" if score< -1 else "HOLD"
    return sig

def generate_option_notifications(tickers):
    out = {}
    for t in tickers:
        try:
            tk   = yf.Ticker(t)
            exps = tk.options
            if not exps: continue
            chain = tk.option_chain(exps[0])
            call_vol = chain.calls["volume"].sum()
            put_vol  = chain.puts["volume"].sum()
            if call_vol+put_vol==0: continue
            ratio = call_vol/put_vol if put_vol else np.inf
            if ratio>1.2: out[t] = f"CALL ALERT ({ratio:.2f}×)"
            elif ratio<0.8: out[t] = f"PUT ALERT  ({ratio:.2f}×)"
        except:
            pass
    return out

def send_alerts(signals, opts):
    lines = []
    for t, s in signals.items():
        l = f"{t}: {s}"
        if t in opts: l += " | "+opts[t]
        lines.append(l)
    msg = "\n".join(lines)
    if bot:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    else:
        print(msg)

def main():
    feats = build_feature_matrix(TICKERS)
    feats_exp, pca_model, umap_model = apply_pca_umap(feats)
    print("Features + embeddings:\n", feats_exp)
    sigs = generate_signals(feats_exp)
    opt  = generate_option_notifications(OPTION_TICKERS)
    send_alerts(sigs, opt)

if __name__=="__main__":
    main()
    
