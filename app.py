import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(layout="wide", page_title="Stock Predictor")

# Title
st.title("📈 Stock Predictor Demo")

# Load dataset
uploaded = st.file_uploader("Upload all_stocks_5yr.csv", type="csv")
if uploaded is None:
    df = pd.read_csv('/content/all_stocks_5yr.csv')
else:
    df = pd.read_csv(uploaded)

# Select ticker
tickers = df['Name'].unique().tolist()
ticker = st.selectbox("Select ticker", tickers)

# Filter data for the selected ticker
df_t = df[df['Name'] == ticker].copy()
df_t['date'] = pd.to_datetime(df_t['date'])
df_t = df_t.sort_values('date').reset_index(drop=True)

# Feature engineering
df_t['ma10'] = df_t['close'].rolling(10).mean()
df_t['ma50'] = df_t['close'].rolling(50).mean()
df_t['prev_close'] = df_t['close'].shift(1)
df_t = df_t.dropna().reset_index(drop=True)

# Candlestick chart
st.subheader(f"{ticker} — Candlestick Chart with MA10 & MA50")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_t['date'], open=df_t['open'], high=df_t['high'],
    low=df_t['low'], close=df_t['close'], name='Candlestick'
))
fig.add_trace(go.Scatter(x=df_t['date'], y=df_t['ma10'], name='MA10', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_t['date'], y=df_t['ma50'], name='MA50', line=dict(color='orange')))
st.plotly_chart(fig, use_container_width=True)

# RandomForest Prediction (Demo)
st.subheader("RandomForest Prediction (Demo)")

feats = ['open','high','low','volume','ma10','ma50','prev_close']
X = df_t[feats].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load pre-trained model if exists
model_path = Path('models') / f'rf_{ticker.lower()}.joblib'
if model_path.exists():
    rf = joblib.load(model_path)
    preds = rf.predict(X_scaled[-30:])
    
    df_plot = df_t.iloc[-30:].copy()
    df_plot['pred'] = preds
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], name='Actual'))
    fig2.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['pred'], name='Predicted'))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No trained model found. Please run train.py first to generate RandomForest models.")
