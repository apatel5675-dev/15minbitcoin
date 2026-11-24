"""
Crypto Market Research Dashboard
Educational market-trend classification and historical model evaluation.
The trend agreement score is NOT a probability of a future price move or a betting signal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

st.set_page_config(page_title="Crypto Market Research", layout="wide")
st.title("Crypto Market Research Dashboard")
st.caption("Educational research only. No Kalshi connection or betting recommendations.")

st.sidebar.header("Market data")
symbol = st.sidebar.selectbox("Main asset", ["BTC/USD", "ETH/USD", "SOL/USD"])
timeframe = st.sidebar.selectbox("Candle interval", ["5m", "15m", "1h", "4h"], index=1)
limit = st.sidebar.slider("Candles to load", 300, 1500, 1000, step=100)

@st.cache_data(ttl=60)
def load_ohlcv(pair, interval, count):
    exchange = ccxt.kraken({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(pair, timeframe=interval, limit=count)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def add_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
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
    df["ema_gap"] = df["ema20"] / df["ema50"] - 1
    df["macd_gap"] = df["macd"] - df["macd_signal"]
    return df

try:
    data = add_indicators(load_ohlcv(symbol, timeframe, limit))
except Exception as exc:
    st.error(f"Could not load Kraken market data: {exc}")
    st.info("Try refreshing later. The hosting service must allow outbound requests to Kraken.")
    st.stop()

latest, previous = data.iloc[-1], data.iloc[-2]
move = (latest["close"] / previous["close"] - 1) * 100

# Indicator agreement describes the current technical picture only.
votes = []
if pd.notna(latest["ema20"]) and pd.notna(latest["ema50"]):
    votes.append(1 if latest["close"] > latest["ema20"] and latest["ema20"] > latest["ema50"]
                 else -1 if latest["close"] < latest["ema20"] and latest["ema20"] < latest["ema50"]
                 else 0)
if pd.notna(latest["macd_gap"]):
    votes.append(1 if latest["macd_gap"] > 0 else -1 if latest["macd_gap"] < 0 else 0)
if pd.notna(latest["rsi"]):
    votes.append(1 if latest["rsi"] > 55 else -1 if latest["rsi"] < 45 else 0)
if pd.notna(latest["return_3"]):
    votes.append(1 if latest["return_3"] > 0 else -1 if latest["return_3"] < 0 else 0)

up_votes = sum(v == 1 for v in votes)
down_votes = sum(v == -1 for v in votes)
neutral_votes = sum(v == 0 for v in votes)
if up_votes > down_votes:
    trend = "UP"
    agreement = up_votes / max(len(votes), 1) * 100
elif down_votes > up_votes:
    trend = "DOWN"
    agreement = down_votes / max(len(votes), 1) * 100
else:
    trend = "NEUTRAL"
    agreement = max(up_votes, down_votes, neutral_votes) / max(len(votes), 1) * 100

st.subheader("Current market snapshot")
a, b, c, d = st.columns(4)
a.metric("Latest candle close", f"${latest['close']:,.2f}", f"{move:+.3f}%")
a2 = b
a2.metric("RSI (14)", f"{latest['rsi']:.1f}" if pd.notna(latest["rsi"]) else "—")
c.metric("MACD", f"{latest['macd']:.3f}" if pd.notna(latest["macd"]) else "—")
d.metric("20-candle volatility", f"{latest['volatility20']*100:.3f}%" if pd.notna(latest["volatility20"]) else "—")

st.header("Current trend classification")
trend_col, agreement_col = st.columns(2)
trend_col.metric("Indicator-based trend", trend)
agreement_col.metric("Indicator agreement", f"{agreement:.0f}%")
st.caption(
    "Agreement means the share of available indicators voting for the displayed current trend. "
    "It is not a forecast probability, confidence that the next candle will move that way, "
    "or an estimate of a Kalshi contract outcome."
)
st.write(f"Indicator votes — UP: {up_votes} | DOWN: {down_votes} | NEUTRAL: {neutral_votes}")

with st.expander("See how the indicators voted"):
    st.write(f"**EMA trend:** {'UP' if votes and len(votes) > 0 and latest['close'] > latest['ema20'] > latest['ema50'] else 'DOWN' if latest['close'] < latest['ema20'] < latest['ema50'] else 'Mixed'}")
    st.write(f"**MACD:** {'Positive' if latest['macd_gap'] > 0 else 'Negative'} relative to its signal line.")
    st.write(f"**RSI:** {latest['rsi']:.1f} (above 55 is counted as an upward momentum vote; below 45 as downward).")
    st.write(f"**Three-candle return:** {latest['return_3']*100:+.3f}%.")

st.subheader(f"{symbol} historical chart ({timeframe})")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    row_heights=[0.60, 0.20, 0.20],
                    subplot_titles=("Candles and moving averages", "RSI (14)", "Volume"))
fig.add_trace(go.Candlestick(x=data["timestamp"], open=data["open"], high=data["high"],
                             low=data["low"], close=data["close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema20"], name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["ema50"], name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["rsi"], name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", row=2, col=1)
fig.add_trace(go.Bar(x=data["timestamp"], y=data["volume"], name="Volume"), row=3, col=1)
fig.update_layout(height=800, xaxis_rangeslider_visible=False, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.header("Historical machine-learning experiment")
st.write(
    "A Random Forest is trained on the earlier 75% of the selected history and tested on "
    "the later 25%. The target is whether the next completed candle closed above the current "
    "close. This is a retrospective classroom experiment, not a live forecast."
)
features = ["rsi", "macd_gap", "ema_gap", "return_1", "return_3",
            "return_6", "volatility20", "volume_ratio"]
ml = data.copy()
ml["next_return"] = ml["close"].shift(-1) / ml["close"] - 1
ml["next_up"] = (ml["next_return"] > 0).astype(int)
ml = ml.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["next_return"]).iloc[:-1]
if len(ml) < 250:
    st.warning("Not enough complete candles for a useful historical test. Load more candles.")
else:
    split = int(len(ml) * 0.75)
    train, test = ml.iloc[:split], ml.iloc[split:]
    model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=8,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(train[features], train["next_up"])
    predicted = model.predict(test[features])
    acc = accuracy_score(test["next_up"], predicted)
    bal = balanced_accuracy_score(test["next_up"], predicted)
    x, y, z = st.columns(3)
    x.metric("Historical test accuracy", f"{acc*100:.1f}%")
    y.metric("Balanced accuracy", f"{bal*100:.1f}%")
    z.metric("Held-out candles", f"{len(test):,}")
    st.caption(f"Training rows: {len(train):,} | Test rows: {len(test):,} | Test starts: {test['timestamp'].iloc[0]}")
    matrix = confusion_matrix(test["next_up"], predicted, labels=[0, 1])
    st.dataframe(pd.DataFrame(matrix, index=["Actual DOWN", "Actual UP"],
                              columns=["Classified DOWN", "Classified UP"]), use_container_width=True)
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    imp_fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
    imp_fig.update_layout(height=350, title="Relative feature importance")
    st.plotly_chart(imp_fig, use_container_width=True)
    st.warning(
        "Historical accuracy is not a future probability. Results can change with the time period, "
        "market regime, data quality, and model settings. Do not interpret this as a Kalshi signal."
    )

st.header("Cross-crypto historical comparison")
try:
    ret = {}
    for pair in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        tmp = load_ohlcv(pair, timeframe, limit)
        ret[pair] = tmp.set_index("timestamp")["close"].pct_change()
    st.write("Correlation of candle returns")
    st.dataframe(pd.DataFrame(ret).dropna().corr().round(3), use_container_width=True)
except Exception as exc:
    st.warning(f"Cross-asset data could not be loaded: {exc}")

st.subheader("Recent candles")
st.dataframe(data[["timestamp", "open", "high", "low", "close", "volume",
                   "rsi", "macd", "ema20", "ema50"]].tail(20).sort_values(
                       "timestamp", ascending=False), use_container_width=True)

st.info(
    "Kalshi is not connected in this educational version. No contract selection, betting "
    "recommendation, or contract-outcome confidence score is provided. Market data: Kraken via CCXT."
)
