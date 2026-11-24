"""
Crypto Market Research Dashboard
Educational use only: historical indicators and out-of-sample model evaluation.
It does not connect to prediction markets or provide live betting/trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, confusion_matrix, brier_score_loss
)

st.set_page_config(page_title="Crypto Market Research", layout="wide")
st.title("Crypto Market Research Dashboard")
st.caption(
    "Educational research prototype. Shows historical market data and tests a model "
    "on past unseen candles. It does not generate live UP/DOWN recommendations."
)

st.sidebar.header("Market data")
symbol = st.sidebar.selectbox("Main asset", ["BTC/USD", "ETH/USD", "SOL/USD"])
timeframe = st.sidebar.selectbox("Candle interval", ["5m", "15m", "1h", "4h"], index=1)
limit = st.sidebar.slider("Candles to load", 300, 1500, 1000, step=100)

@st.cache_data(ttl=60)
def load_ohlcv(pair, interval, count):
    exchange = ccxt.kraken({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(pair, timeframe=interval, limit=count)
    frame = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame

def add_indicators(frame):
    df = frame.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["return_1"] = df["close"].pct_change()
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["volatility20"] = df["return_1"].rolling(20).std()
    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg20"]
    df["ema_gap"] = (df["ema20"] / df["ema50"]) - 1
    df["macd_gap"] = df["macd"] - df["macd_signal"]
    return df

try:
    data = add_indicators(load_ohlcv(symbol, timeframe, limit))
except Exception as exc:
    st.error(f"Could not load Kraken market data: {exc}")
    st.info(
        "This app uses Kraken public OHLCV data. If it is temporarily unavailable, "
        "refresh later or check whether the hosting environment permits outbound API requests."
    )
    st.stop()

latest = data.iloc[-1]
prev = data.iloc[-2]
pct_change = (latest["close"] / prev["close"] - 1) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest candle close", f"${latest['close']:,.2f}", f"{pct_change:+.3f}%")
m2.metric("RSI (14)", f"{latest['rsi']:.1f}" if pd.notna(latest["rsi"]) else "—")
m3.metric("MACD", f"{latest['macd']:.3f}" if pd.notna(latest["macd"]) else "—")
m4.metric("20-candle volatility", f"{latest['volatility20']*100:.3f}%" if pd.notna(latest["volatility20"]) else "—")

st.subheader(f"{symbol} historical price chart — {timeframe}")
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    row_heights=[0.60, 0.20, 0.20],
    subplot_titles=("Candlesticks and moving averages", "RSI (14)", "Volume")
)
fig.add_trace(go.Candlestick(
    x=data["timestamp"], open=data["open"], high=data["high"],
    low=data["low"], close=data["close"], name="Price"
), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema20"], name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema50"], name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["rsi"], name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", row=2, col=1)
fig.add_trace(go.Bar(x=data["timestamp"], y=data["volume"], name="Volume"), row=3, col=1)
fig.update_layout(height=820, xaxis_rangeslider_visible=False, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Indicator context")
notes = []
if latest["close"] > latest["ema20"] and latest["ema20"] > latest["ema50"]:
    notes.append("Price is above EMA 20, and EMA 20 is above EMA 50 in the latest candle.")
elif latest["close"] < latest["ema20"] and latest["ema20"] < latest["ema50"]:
    notes.append("Price is below EMA 20, and EMA 20 is below EMA 50 in the latest candle.")
else:
    notes.append("Price and moving-average alignment is mixed in the latest candle.")
if pd.notna(latest["rsi"]):
    notes.append(f"RSI is {latest['rsi']:.1f}; common reference levels are 30 and 70, not automatic buy/sell rules.")
if pd.notna(latest["macd_gap"]):
    notes.append("MACD is above its signal line." if latest["macd_gap"] > 0 else "MACD is below its signal line.")
if pd.notna(latest["volume_ratio"]):
    notes.append(f"Current volume is {latest['volume_ratio']:.2f} times its recent 20-candle average.")
for item in notes:
    st.write("• " + item)

st.divider()
st.header("Historical machine-learning experiment")
st.write(
    "This section trains a Random Forest on earlier candles and evaluates it on a later "
    "held-out segment. The label describes whether the *next completed candle* closed "
    "above or below the current close. Results are historical test statistics only; "
    "the app intentionally does not produce a live next-candle call."
)

feature_cols = [
    "rsi", "macd_gap", "ema_gap", "return_1", "return_3",
    "return_6", "volatility20", "volume_ratio"
]
model_data = data.copy()
model_data["next_return"] = model_data["close"].shift(-1) / model_data["close"] - 1
model_data["next_up"] = (model_data["next_return"] > 0).astype(int)
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["next_return"])
model_data = model_data.iloc[:-1].copy()

if len(model_data) < 250:
    st.warning("Not enough complete observations for a meaningful chronological test. Load more candles.")
else:
    split = int(len(model_data) * 0.75)
    train = model_data.iloc[:split]
    test = model_data.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=250, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(train[feature_cols], train["next_up"])
    pred = model.predict(test[feature_cols])
    probs = model.predict_proba(test[feature_cols])[:, list(model.classes_).index(1)]

    acc = accuracy_score(test["next_up"], pred)
    bal_acc = balanced_accuracy_score(test["next_up"], pred)
    precision = precision_score(test["next_up"], pred, zero_division=0)
    recall = recall_score(test["next_up"], pred, zero_division=0)
    brier = brier_score_loss(test["next_up"], probs)

    a, b, c, d = st.columns(4)
    a.metric("Held-out accuracy", f"{acc*100:.1f}%")
    b.metric("Balanced accuracy", f"{bal_acc*100:.1f}%")
    c.metric("Precision (UP class)", f"{precision*100:.1f}%")
    d.metric("Brier score", f"{brier:.3f}")

    st.caption(
        f"Training observations: {len(train):,} | Held-out observations: {len(test):,} | "
        f"Test period: {test['timestamp'].iloc[0]} through {test['timestamp'].iloc[-1]}"
    )
    st.write(
        "Accuracy is the share of test candles classified correctly. Balanced accuracy "
        "accounts for class imbalance. Precision measures how often the model's historical "
        "UP classifications were correct. Brier score measures probability error (lower is better)."
    )

    cm = confusion_matrix(test["next_up"], pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm, index=["Actual DOWN", "Actual UP"],
        columns=["Predicted DOWN", "Predicted UP"]
    )
    st.subheader("Held-out confusion matrix")
    st.dataframe(cm_df, use_container_width=True)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    st.subheader("Which historical features mattered most to this model?")
    imp_fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
    imp_fig.update_layout(height=380, xaxis_title="Relative feature importance", yaxis_title="Feature")
    st.plotly_chart(imp_fig, use_container_width=True)

    st.warning(
        "A historical accuracy percentage is not the probability that a future move will occur. "
        "Small samples, changing market conditions, fees, slippage, and data quality can make "
        "real-world performance substantially different. This experiment is not a betting signal."
    )

st.divider()
st.header("Compare crypto market history")
st.write("The table compares historical candle returns for major crypto assets.")
try:
    assets = ["BTC/USD", "ETH/USD", "SOL/USD"]
    returns = {}
    for asset in assets:
        frame = load_ohlcv(asset, timeframe, limit)
        returns[asset] = frame.set_index("timestamp")["close"].pct_change()
    return_frame = pd.DataFrame(returns).dropna()
    st.subheader("Correlation of candle returns")
    st.dataframe(return_frame.corr().round(3), use_container_width=True)
    st.caption("Correlation describes how returns moved together over the selected historical sample; it does not establish causation.")
except Exception as exc:
    st.warning(f"Cross-asset comparison could not be loaded: {exc}")

st.subheader("Recent market data")
st.dataframe(
    data[["timestamp", "open", "high", "low", "close", "volume",
          "rsi", "macd", "ema20", "ema50"]].tail(25).sort_values("timestamp", ascending=False),
    use_container_width=True
)

st.info(
    "Kalshi note: this version does not connect to Kalshi or display contract prices. "
    "It focuses on public exchange data and historical model evaluation, not prediction-market participation."
)
st.caption("Data source: Kraken public OHLCV market data through CCXT. Educational use only.")
