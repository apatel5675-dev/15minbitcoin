"""
Educational cryptocurrency market dashboard.
Displays historical market data and technical indicators for research.
It does NOT predict the next candle or generate betting/trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Crypto Market Research", layout="wide")
st.title("Crypto Market Research Dashboard")
st.caption(
    "Educational research tool: historical prices, indicators, and market context. "
    "Not financial advice and not a prediction or betting-signal service."
)

st.sidebar.header("Data settings")
symbol = st.sidebar.selectbox("Main asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
timeframe = st.sidebar.selectbox("Candle interval", ["5m", "15m", "1h", "4h"], index=1)
limit = st.sidebar.slider("Candles to load", 100, 1000, 500, step=50)

@st.cache_data(ttl=60)
def load_ohlcv(symbol, timeframe, limit):
    exchange = ccxt.binance({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def add_indicators(df):
    df = df.copy()
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
    df["return_pct"] = df["close"].pct_change() * 100
    df["volatility"] = df["return_pct"].rolling(20).std()
    df["volume_avg20"] = df["volume"].rolling(20).mean()
    return df

try:
    data = add_indicators(load_ohlcv(symbol, timeframe, limit))
except Exception as exc:
    st.error(f"Could not load exchange data: {exc}")
    st.info("Check your internet connection or try again in a moment.")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2]
change = (latest["close"] / previous["close"] - 1) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"${latest['close']:,.2f}", f"{change:+.3f}%")
c2.metric("RSI (14)", f"{latest['rsi']:.1f}" if pd.notna(latest["rsi"]) else "—")
c3.metric("MACD", f"{latest['macd']:.3f}" if pd.notna(latest["macd"]) else "—")
c4.metric("20-candle volatility", f"{latest['volatility']:.3f}%" if pd.notna(latest["volatility"]) else "—")

st.subheader(f"{symbol} price history ({timeframe} candles)")
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    row_heights=[0.60, 0.20, 0.20],
    subplot_titles=("Candlesticks and moving averages", "RSI (14)", "Volume")
)
fig.add_trace(
    go.Candlestick(
        x=data["timestamp"], open=data["open"], high=data["high"],
        low=data["low"], close=data["close"], name="Price"
    ), row=1, col=1
)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema20"], name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema50"], name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["rsi"], name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", row=2, col=1)
fig.add_trace(go.Bar(x=data["timestamp"], y=data["volume"], name="Volume"), row=3, col=1)
fig.update_layout(height=850, xaxis_rangeslider_visible=False, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Indicator guide")
st.markdown(
    """
    - **RSI:** A momentum oscillator. Values near 70 or 30 are commonly watched,
      but they are not reliable standalone reversal signals.
    - **MACD:** Compares moving averages to describe momentum changes.
    - **EMA 20 / EMA 50:** Smoothed price averages used to study trend context.
    - **Volume:** Shows how much of the asset traded during each candle.
    - **Volatility:** Measures how much recent percentage returns have varied.
    """
)

# Descriptive context only; intentionally no next-candle forecast or UP/DOWN signal.
trend_notes = []
if latest["close"] > latest["ema20"] and latest["ema20"] > latest["ema50"]:
    trend_notes.append("Recent price and moving-average alignment is consistent with a positive short-term trend.")
elif latest["close"] < latest["ema20"] and latest["ema20"] < latest["ema50"]:
    trend_notes.append("Recent price and moving-average alignment is consistent with a negative short-term trend.")
else:
    trend_notes.append("Moving averages are mixed; the recent trend is not clearly aligned.")

if pd.notna(latest["rsi"]):
    if latest["rsi"] >= 70:
        trend_notes.append("RSI is in a commonly watched high range; this does not by itself mean price must fall.")
    elif latest["rsi"] <= 30:
        trend_notes.append("RSI is in a commonly watched low range; this does not by itself mean price must rise.")
    else:
        trend_notes.append("RSI is between the commonly watched 30 and 70 levels.")

if pd.notna(latest["macd"]) and pd.notna(latest["macd_signal"]):
    trend_notes.append(
        "MACD is above its signal line." if latest["macd"] > latest["macd_signal"]
        else "MACD is below its signal line."
    )

st.subheader("Current historical-indicator context")
for note in trend_notes:
    st.write("• " + note)

st.subheader("Compare major crypto assets")
assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
comparison = {}
try:
    for asset in assets:
        frame = load_ohlcv(asset, timeframe, limit)
        frame["return_pct"] = frame["close"].pct_change() * 100
        comparison[asset] = frame.set_index("timestamp")["return_pct"]
    returns = pd.DataFrame(comparison).dropna()
    st.write("Correlation of candle returns over the loaded period")
    st.dataframe(returns.corr().round(3), use_container_width=True)
except Exception as exc:
    st.warning(f"Cross-asset comparison was unavailable: {exc}")

st.subheader("Recent data")
st.dataframe(
    data[["timestamp", "open", "high", "low", "close", "volume",
          "rsi", "macd", "ema20", "ema50"]].tail(20).sort_values("timestamp", ascending=False),
    use_container_width=True
)

st.caption(
    "Data source: Binance public market candles via CCXT. "
    "Data can be delayed, incomplete, or unavailable. This app does not connect to Kalshi, "
    "recommend bets, or forecast future price direction."
)
