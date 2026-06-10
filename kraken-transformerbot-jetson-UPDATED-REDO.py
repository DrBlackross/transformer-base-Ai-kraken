#### THIS BOT WILL NOT SEE A PROFIT TILL KRAKEN DROPS THEIR MAKER/TAKER FEES
## changed to 
## 30- Day Volume (USD)     Maker      Taker
##     $0+	                0.25%	   0.40%
#### which sucks, i tweeked it enough to a point i was about to post this updated script here and that happened
#### added back testing btw
# Backtest with new settings
## python kraken-transformerbot-jetson-UPDATED-REDO.py --mode backtest --days 90

# Train the AI model
## python kraken-transformerbot-jetson-UPDATED-REDO.py --mode train --days 90

# Live trading (paper or real based on .env)
## python kraken-transformerbot-jetson-UPDATED-REDO.py --mode live

#### Need an .env file in the root of the script, format is....
# LIVE_TRADING=true
# KRAKEN_API_KEY=KRAKEN_KEY_HERE
# KRAKEN_API_SECRET=KRAKEN_SECRET_HERE
# TRADING_PAIR=DOGEUSDT
# DOGEUSDT or XBTUSDT
# USE_REASONING_LM=true
# DECISION_INTERVAL_SECONDS=180
# DISCORD_WEBHOOK_URL=PUT DISCORD WEBHOOK HERE 
#### the discord webhook was nice lol


#!/usr/bin/env python3

import os
import time
import json
import shutil
import warnings
import threading
from datetime import datetime, timedelta, timezone
import logging
import sys
import numpy as np
import pandas as pd
import requests
import joblib
import torch
import talib
from dotenv import load_dotenv
from flask import Flask, render_template_string, jsonify
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from krakenex import API as KrakenEx
from pykrakenapi import KrakenAPI
import pytz
from tzlocal import get_localzone

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Kraken Transformer Trading Bot")
parser.add_argument(
    "--mode",
    type=str,
    choices=["live", "backtest", "train"],
    default="live",
    help="Run mode: live (real trading), backtest (simulate), or train (retrain only)",
)
parser.add_argument(
    "--days", type=int, default=90, help="Number of days for backtest or training"
)
parser.add_argument(
    "--verbose", action="store_true", help="Print detailed backtest results"
)
args = parser.parse_args()

# ======================
# DISCORD WEBHOOK NOTIFICATIONS
# ======================

DISCORD_WEBHOOK_ENABLED = (
    os.getenv("DISCORD_WEBHOOK_ENABLED", "false").lower() == "true"
)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_message(message, title=None):
    """Send a message to Discord via webhook"""
    if not DISCORD_WEBHOOK_ENABLED or not DISCORD_WEBHOOK_URL:
        return False

    try:
        payload = {"content": message}

        if title:
            payload["embeds"] = [
                {
                    "title": title,
                    "description": message,
                    "color": (
                        0x00FF00
                        if "BUY" in message.upper()
                        else 0xFF4444 if "SELL" in message.upper() else 0xFFFF00
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
            payload.pop("content")

        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"⚠️ Discord error: {e}")
        return False


def send_trade_alert(trade_type, volume, price, txid=None, pnl=None):
    """Send formatted trade alert to Discord"""
    if not DISCORD_WEBHOOK_ENABLED:
        return

    emoji = "🟢" if trade_type.upper() == "BUY" else "🔴"

    message = f"""{emoji} **{trade_type.upper()} ORDER EXECUTED** {emoji}

**Pair:** {PAIR}
**Amount:** {volume:.4f} {CRYPTO_NAME}
**Price:** ${price:.6f}
**Total:** ${volume * price:.2f}"""

    if txid:
        message += f"\n**TXID:** `{txid[:12] if txid else 'N/A'}...`"

    if pnl is not None:
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        message += f"\n\n{pnl_emoji} **Trade P/L:** ${pnl:.2f}"

    # Add overall portfolio PnL
    with app_state_lock:
        overall_pnl = app_state.get("pnl", 0)
        overall_pnl_pct = app_state.get("pnl_percent", 0)

    overall_emoji = "✅" if overall_pnl >= 0 else "❌"
    message += f"\n\n{overall_emoji} **Portfolio P/L:** ${overall_pnl:.2f} ({overall_pnl_pct:+.2f}%)"

    send_discord_message(message, f"{trade_type.upper()} Alert")


def send_status_update(
    crypto_balance,
    usdt_balance,
    portfolio_value,
    rsi,
    price,
    pnl=None,
    pnl_percent=None,
):
    """Send periodic status update to Discord"""
    if not DISCORD_WEBHOOK_ENABLED:
        return

    # Calculate PnL if not provided
    if pnl is None or pnl_percent is None:
        # Try to get from app_state
        with app_state_lock:
            pnl = app_state.get("pnl", 0)
            pnl_percent = app_state.get("pnl_percent", 0)

    pnl_emoji = "✅" if pnl >= 0 else "❌"
    pnl_color = "green" if pnl >= 0 else "red"

    message = f"""**📊 Bot Status Update**

**{CRYPTO_NAME}:** {crypto_balance:.4f}
**USDT:** ${usdt_balance:.2f}
**Portfolio:** ${portfolio_value:.2f}
**RSI:** {rsi:.1f}
**Price:** ${price:.6f}

{pnl_emoji} **P/L:** ${pnl:.2f} ({pnl_percent:+.2f}%)"""

    send_discord_message(message, "Trading Bot Status")


# ======================
# USER CONFIGURABLE SETTINGS
# ======================

LIVE_TRADING = True
print(f"\n{'='*60}")
print(f"🔴🔴🔴 LIVE_TRADING = {LIVE_TRADING} 🔴🔴🔴")
print(f"🔴🔴🔴 REAL MONEY IS AT RISK! 🔴🔴🔴")
print(f"{'='*60}\n")

USE_REASONING_LM = os.getenv("USE_REASONING_LM", "true").lower() == "true"
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Select trading pair
TRADING_PAIR = os.getenv("TRADING_PAIR", "DOGEUSDT").upper()

# --- Trading Pair Specific Settings ---
if TRADING_PAIR == "DOGEUSDT":
    MODEL_PATH_BASE = "./doge_transformer_model"
    SCALER_PATH_BASE = "./standard_scaler"
    CRYPTO_NAME = "DOGE"
    CRYPTO_ASSET_KEY = "XXDG"
    INITIAL_USDT_BALANCE = None
    INITIAL_CRYPTO_BALANCE = None
    MIN_TRADE_AMOUNT = 5.0
    CRYPTO_DECIMALS = 4
    DEFAULT_ATR_MULTIPLIER = 0.05
    BEST_MODEL_PATH = f"{MODEL_PATH_BASE}_best.pth"
    BEST_SCALER_PATH = f"{SCALER_PATH_BASE}_best.pkl"
    BEST_MULTIPLIER_PATH = f"{MODEL_PATH_BASE}_best_multiplier.txt"
    BEST_LEARNING_RATE_PATH = f"{MODEL_PATH_BASE}_best_learning_rate.txt"

elif TRADING_PAIR == "XBTUSDT":
    MODEL_PATH_BASE = "./btc_transformer_model"
    SCALER_PATH_BASE = "./standard_scaler"
    CRYPTO_NAME = "BTC"
    CRYPTO_ASSET_KEY = "XXBT"
    INITIAL_USDT_BALANCE = None
    INITIAL_CRYPTO_BALANCE = None
    MIN_TRADE_AMOUNT = 0.0001
    CRYPTO_DECIMALS = 6
    DEFAULT_ATR_MULTIPLIER = 0.05
    BEST_MODEL_PATH = f"{MODEL_PATH_BASE}_best.pth"
    BEST_SCALER_PATH = f"{SCALER_PATH_BASE}_best.pkl"
    BEST_MULTIPLIER_PATH = f"{MODEL_PATH_BASE}_best_multiplier.txt"
    BEST_LEARNING_RATE_PATH = f"{MODEL_PATH_BASE}_best_learning_rate.txt"

else:
    raise ValueError("Invalid TRADING_PAIR. Use DOGEUSDT or XBTUSDT.")

# --- Trading Parameters ---
PAIR = TRADING_PAIR
INTERVAL = 5
LOOKBACK_DAYS_TRAINING = int(os.getenv("LOOKBACK_DAYS_TRAINING", "30"))
LOOKBACK_DAYS_WINDOW = int(os.getenv("LOOKBACK_DAYS_WINDOW", "2"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "12"))
DECISION_INTERVAL_SECONDS = 120
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "6"))
last_trade_price = 0

# Trading Fees Configuration
EXCHANGE_FEES = {"maker": 0.0025, "taker": 0.0040, "default": 0.0040}

# Aggressive Trading Settings
TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", "1"))  # was 0.95
MAX_NOTIONAL_PER_TRADE = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "50"))

# --- Model Training Parameters ---
NUM_TRAINING_RUNS = 2
NUM_TRAIN_EPOCHS = 10
DROPOUT_RATE = 0.3

# --- OPTIMIZATION PARAMETERS ---
ATR_MULTIPLIERS_TO_TEST = [0.01, 0.02, 0.03, 0.04, 0.05]
LEARNING_RATES_TO_TEST = [1e-4, 5e-4, 1e-3]

# --- TECHNICAL INDICATOR CONFIGURATION ---
INDICATOR_CONFIG = {
    "BASE_FEATURES": ["open", "high", "low", "close", "volume"],
    "RSI": {"enabled": True, "length": 7, "overbought": 72, "oversold": 28},
    "MACD": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
    "BBANDS": {"enabled": True, "length": 20, "std": 2},
    "OBV": {"enabled": True},
    "ADX": {"enabled": True, "length": 14},
    "CCI": {"enabled": True, "length": 20, "c": 0.015},
    "ATR": {"enabled": True, "length": 14},
    "VWAP": {"enabled": False},
    "NATR": {"enabled": True, "length": 14},
    "TRIX": {"enabled": True, "length": 15, "signal": 9},
    "STOCH": {"enabled": True, "k": 14, "d": 3, "smooth_k": 3},
    "EMA": {"enabled": True, "lengths": [9, 21, 50]},
}


def build_features_list(cfg):
    feats = list(cfg["BASE_FEATURES"])
    if cfg["RSI"]["enabled"]:
        feats.append(f"RSI_{cfg['RSI']['length']}")
    if cfg["MACD"]["enabled"]:
        macd = cfg["MACD"]
        feats.extend(
            [
                f"MACD_{macd['fast']}_{macd['slow']}_{macd['signal']}",
                f"MACDh_{macd['fast']}_{macd['slow']}_{macd['signal']}",
                f"MACDs_{macd['fast']}_{macd['slow']}_{macd['signal']}",
            ]
        )
    if cfg["BBANDS"]["enabled"]:
        bb = cfg["BBANDS"]
        feats.extend(
            [
                f"BBL_{bb['length']}_{bb['std']}",
                f"BBM_{bb['length']}_{bb['std']}",
                f"BBU_{bb['length']}_{bb['std']}",
            ]
        )
    if cfg["OBV"]["enabled"]:
        feats.append("OBV")
    if cfg["ADX"]["enabled"]:
        feats.append(f"ADX_{cfg['ADX']['length']}")
    if cfg["CCI"]["enabled"]:
        cci = cfg["CCI"]
        feats.append(f"CCI_{cci['length']}_{cci['c']}")
    if cfg["ATR"]["enabled"]:
        feats.append(f"ATRr_{cfg['ATR']['length']}")
    if cfg["NATR"]["enabled"]:
        feats.append(f"NATR_{cfg['NATR']['length']}")
    if cfg["TRIX"]["enabled"]:
        trix = cfg["TRIX"]
        feats.extend(
            [
                f"TRIX_{trix['length']}_{trix['signal']}",
                f"TRIXs_{trix['length']}_{trix['signal']}",
            ]
        )
    if cfg["STOCH"]["enabled"]:
        st = cfg["STOCH"]
        feats.extend(
            [
                f"STOCHk_{st['k']}_{st['d']}_{st['smooth_k']}",
                f"STOCHd_{st['k']}_{st['d']}_{st['smooth_k']}",
            ]
        )
    if cfg["EMA"]["enabled"]:
        for length in cfg["EMA"]["lengths"]:
            feats.append(f"EMA_{length}")
    return feats


FEATURES_LIST = build_features_list(INDICATOR_CONFIG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using device={DEVICE}")

trained_model = None
final_scaler = None

# --- Kraken API Setup ---
kraken_ex = (
    KrakenEx(key=KRAKEN_API_KEY, secret=KRAKEN_API_SECRET)
    if KRAKEN_API_KEY and KRAKEN_API_SECRET
    else KrakenEx()
)
kraken_api = KrakenAPI(kraken_ex)

# --- Global State for Web Interface ---
app_state = {
    "training_progress": 0,
    "training_status": "Not Training",
    "is_training": False,
    "best_multiplier": "N/A",
    "best_learning_rate": "N/A",
    "crypto_balance": 0.0,
    "usdt_balance": 0.0,
    "portfolio_value": 0.0,
    "pnl": 0.0,
    "pnl_percent": 0.0,
    "last_decision_time": None,
    "trade_history": [],
    "current_multiplier": "N/A",
    "current_learning_rate": "N/A",
    "current_run": 0,
    "total_runs": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "last_training_time": None,
    "next_training_time": None,
    "optimization_phase": "Not Started",
    "claude_analysis": {},
    "target_price_info": None,
}

app_state_lock = threading.Lock()

# --- Flask App Setup ---
app = Flask(__name__)

import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Simple price cache
_last_price = None
_last_price_time = None


def get_current_price():
    """Get current price from multiple sources"""
    global _last_price, _last_price_time

    now = datetime.now(timezone.utc)

    if _last_price is not None and _last_price_time is not None:
        if (now - _last_price_time).total_seconds() < 10:
            return _last_price

    try:
        symbol = PAIR.upper()
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            price = float(response.json()["price"])
            _last_price = price
            _last_price_time = now
            return price
    except:
        pass

    try:
        coin = "dogecoin" if "DOGE" in PAIR else "bitcoin"
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            price = float(response.json()[coin]["usd"])
            _last_price = price
            _last_price_time = now
            return price
    except:
        pass

    return _last_price if _last_price is not None else 0.09


def get_live_balance():
    """Fetch REAL balance from Kraken"""
    try:
        balance_data = kraken_ex.query_private("Balance")
        balances = balance_data["result"]

        crypto_code = "XXDG" if CRYPTO_NAME == "DOGE" else "XXBT"
        crypto_balance = float(balances.get(crypto_code, 0))

        usdt_balance = 0.0
        for code in ["USDT", "USDT.HOLD", "ZUSD"]:
            if code in balances:
                usdt_balance = float(balances[code])
                break

        return crypto_balance, usdt_balance
    except Exception as e:
        print(f"⚠️ Failed to get live balance: {e}")
        return None, None


def add_indicators(df):
    df = df.copy()
    if df.empty:
        return df

    if INDICATOR_CONFIG["RSI"]["enabled"]:
        l = INDICATOR_CONFIG["RSI"]["length"]
        df[f"RSI_{l}"] = talib.RSI(df["close"], timeperiod=l)

    if INDICATOR_CONFIG["MACD"]["enabled"]:
        c = INDICATOR_CONFIG["MACD"]
        macd, macdh, macds = talib.MACD(
            df["close"],
            fastperiod=c["fast"],
            slowperiod=c["slow"],
            signalperiod=c["signal"],
        )
        df[f"MACD_{c['fast']}_{c['slow']}_{c['signal']}"] = macd
        df[f"MACDh_{c['fast']}_{c['slow']}_{c['signal']}"] = macdh
        df[f"MACDs_{c['fast']}_{c['slow']}_{c['signal']}"] = macds

    if INDICATOR_CONFIG["BBANDS"]["enabled"]:
        c = INDICATOR_CONFIG["BBANDS"]
        upper, middle, lower = talib.BBANDS(
            df["close"], timeperiod=c["length"], nbdevup=c["std"], nbdevdn=c["std"]
        )
        df[f"BBU_{c['length']}_{c['std']}"] = upper
        df[f"BBM_{c['length']}_{c['std']}"] = middle
        df[f"BBL_{c['length']}_{c['std']}"] = lower

    if INDICATOR_CONFIG["OBV"]["enabled"]:
        df["OBV"] = talib.OBV(df["close"], df["volume"])

    if INDICATOR_CONFIG["ADX"]["enabled"]:
        l = INDICATOR_CONFIG["ADX"]["length"]
        df[f"ADX_{l}"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=l)

    if INDICATOR_CONFIG["CCI"]["enabled"]:
        c = INDICATOR_CONFIG["CCI"]
        df[f"CCI_{c['length']}_{c['c']}"] = talib.CCI(
            df["high"], df["low"], df["close"], timeperiod=c["length"]
        )

    if INDICATOR_CONFIG["ATR"]["enabled"]:
        l = INDICATOR_CONFIG["ATR"]["length"]
        atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=l)
        df[f"ATRr_{l}"] = atr / df["close"]

    if INDICATOR_CONFIG["NATR"]["enabled"]:
        l = INDICATOR_CONFIG["NATR"]["length"]
        df[f"NATR_{l}"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=l)

    if INDICATOR_CONFIG["TRIX"]["enabled"]:
        c = INDICATOR_CONFIG["TRIX"]
        trix = talib.TRIX(df["close"], timeperiod=c["length"])
        df[f"TRIX_{c['length']}_{c['signal']}"] = trix
        df[f"TRIXs_{c['length']}_{c['signal']}"] = talib.SMA(
            trix, timeperiod=c["signal"]
        )

    if INDICATOR_CONFIG["STOCH"]["enabled"]:
        c = INDICATOR_CONFIG["STOCH"]
        k, d = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=c["k"],
            slowk_period=c["smooth_k"],
            slowd_period=c["d"],
        )
        df[f"STOCHk_{c['k']}_{c['d']}_{c['smooth_k']}"] = k
        df[f"STOCHd_{c['k']}_{c['d']}_{c['smooth_k']}"] = d

    if INDICATOR_CONFIG["EMA"]["enabled"]:
        for l in INDICATOR_CONFIG["EMA"]["lengths"]:
            df[f"EMA_{l}"] = talib.EMA(df["close"], timeperiod=l)

    df.dropna(inplace=True)

    print(f"[debug] DataFrame columns: {list(df.columns)}")

    return df


def fetch_ohlc(pair, interval, lookback_days):
    print(f"[fetch_ohlc] Fetching {lookback_days} days of data...")
    time.sleep(2)
    since = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
    )
    try:
        ohlc, _ = kraken_api.get_ohlc_data(pair, interval=interval, since=since)
        if ohlc is None or ohlc.empty:
            return pd.DataFrame()
        ohlc = ohlc.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        ohlc.index = pd.to_datetime(ohlc.index, unit="s", utc=True)
        return ohlc[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"[fetch_ohlc] Error: {e}")
        return pd.DataFrame()


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_len):
        self.features = np.asarray(features)
        self.targets = np.asarray(targets)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return {
            "input_ids": torch.tensor(x, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
        }


def build_model(input_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(DROPOUT_RATE),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),  # ← MUST be 1 for regression
    )


def prepare_ai_features(df, scaler, seq_len):
    """Prepare features for AI model prediction"""
    try:
        # Get the last seq_len rows
        latest_data = df[FEATURES_LIST].tail(seq_len)
        if len(latest_data) < seq_len:
            return None

        # Scale the features
        scaled_features = scaler.transform(latest_data.values)

        # Convert to tensor [batch, seq_len, features]
        features_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(DEVICE)
        return features_tensor
    except Exception as e:
        print(f"⚠️ AI feature preparation error: {e}")
        return None


def build_train_val_data_from_frame(df):
    X = df[FEATURES_LIST].copy()
    y = df["close"].pct_change().shift(-1).dropna()
    X = X.loc[y.index]
    scaler = StandardScaler()
    scaler.fit(X.values)
    X_scaled = scaler.transform(X.values)
    split = int(len(X_scaled) * 0.8)
    train_ds = TimeSeriesDataset(X_scaled[:split], y.values[:split], SEQUENCE_LENGTH)
    val_ds = TimeSeriesDataset(X_scaled[split:], y.values[split:], SEQUENCE_LENGTH)
    return train_ds, val_ds, scaler


def train_single_model(train_ds, val_ds, scaler, learning_rate, run_idx):
    print(f"[train_single] Starting training run {run_idx} with LR={learning_rate:.2e}")

    # Get a sample batch to determine input dimension
    sample_batch = train_ds[0]
    sample_x = sample_batch["input_ids"]  # Shape: [seq_len, features]
    input_dim = sample_x.shape[1]  # Number of features
    print(f"[train_single] Input dimension: {input_dim}")

    model = build_model(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(NUM_TRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for i in range(len(train_ds)):
            batch = train_ds[i]
            x = batch["input_ids"].to(DEVICE)  # Shape: [seq_len, features]
            y = batch["labels"].float().view(1, 1).to(DEVICE)

            # For a sequence model, we need to aggregate the sequence
            # Option 1: Use the last output (most common)
            # Option 2: Average across sequence (simpler)
            # Let's use the last element of the sequence for prediction
            x_last = x[-1:].to(DEVICE)  # Take only the last time step [1, features]

            pred = model(x_last)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_ds) if len(train_ds) > 0 else 0
        print(
            f"[train_single] Epoch {epoch+1}/{NUM_TRAIN_EPOCHS} - Loss: {avg_loss:.6f}"
        )

    # Save model
    model_path = f"./train_run_{run_idx}/model.pth"
    scaler_path = f"./train_run_{run_idx}/scaler.pkl"
    os.makedirs(f"./train_run_{run_idx}", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    # Simple validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(len(val_ds)):
            batch = val_ds[i]
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].float().view(1, 1).to(DEVICE)
            x_last = x[-1:].to(DEVICE)
            pred = model(x_last)
            val_loss += loss_fn(pred, y).item()
    val_loss /= len(val_ds) if len(val_ds) > 0 else 1

    return model_path, scaler_path, val_loss


def retrain_model():
    global trained_model, final_scaler
    print("[train] Starting retraining...")

    with app_state_lock:
        app_state["is_training"] = True
        app_state["training_status"] = "Training..."
        app_state["training_progress"] = 0
        app_state["optimization_phase"] = "Fetching data..."

    try:
        df = fetch_ohlc(PAIR, INTERVAL, LOOKBACK_DAYS_TRAINING)
        if df.empty:
            print("[train] No data available")
            # Reset training status even on failure
            with app_state_lock:
                app_state["is_training"] = False
                app_state["training_status"] = "Ready"
                app_state["optimization_phase"] = "No data - waiting"
                app_state["training_progress"] = 0
            return

        df = add_indicators(df)
        df.dropna(inplace=True)

        if df.empty:
            print("[train] Not enough data after indicators")
            with app_state_lock:
                app_state["is_training"] = False
                app_state["training_status"] = "Ready"
                app_state["optimization_phase"] = "Insufficient data"
                app_state["training_progress"] = 0
            return

        best_loss = float("inf")
        best_info = None
        total = len(ATR_MULTIPLIERS_TO_TEST) * len(LEARNING_RATES_TO_TEST)
        idx = 0

        for multiplier in ATR_MULTIPLIERS_TO_TEST:
            for lr in LEARNING_RATES_TO_TEST:
                idx += 1
                with app_state_lock:
                    app_state["current_multiplier"] = str(multiplier)
                    app_state["current_learning_rate"] = f"{lr:.2e}"
                    app_state["current_run"] = idx
                    app_state["total_runs"] = total
                    app_state["training_progress"] = int((idx / total) * 100)
                    app_state["optimization_phase"] = (
                        f"Testing ATR={multiplier}, LR={lr:.2e}"
                    )

                train_ds, val_ds, scaler = build_train_val_data_from_frame(df)
                if len(train_ds) < 1 or len(val_ds) < 1:
                    continue

                model_path, scaler_path, val_loss = train_single_model(
                    train_ds, val_ds, scaler, lr, idx
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_info = {
                        "model_path": model_path,
                        "scaler_path": scaler_path,
                        "multiplier": multiplier,
                        "lr": lr,
                    }

        if best_info:
            os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
            shutil.copy(best_info["model_path"], BEST_MODEL_PATH)
            shutil.copy(best_info["scaler_path"], BEST_SCALER_PATH)

            trained_model = build_model(len(FEATURES_LIST)).to(DEVICE)
            trained_model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            )
            trained_model.eval()
            final_scaler = joblib.load(BEST_SCALER_PATH)

            with app_state_lock:
                app_state["best_multiplier"] = str(best_info["multiplier"])
                app_state["best_learning_rate"] = f"{best_info['lr']:.2e}"
                app_state["training_status"] = "Ready"
                app_state["optimization_phase"] = "Complete"
                app_state["training_progress"] = 100

            print(
                f"[train] ✅ Best ATR={best_info['multiplier']} LR={best_info['lr']:.2e} loss={best_loss:.6f}"
            )
        else:
            print("[train] ❌ No valid model found!")
            with app_state_lock:
                app_state["training_status"] = "Ready"
                app_state["optimization_phase"] = "No valid model"
                app_state["training_progress"] = 0

    except Exception as e:
        print(f"[train] ❌ Error: {e}")
        with app_state_lock:
            app_state["training_status"] = "Ready"
            app_state["optimization_phase"] = f"Error: {str(e)[:50]}"
            app_state["training_progress"] = 0
    finally:
        with app_state_lock:
            app_state["is_training"] = False
            # Only update times if training actually completed
            if best_info:
                app_state["last_training_time"] = datetime.now(timezone.utc)
                app_state["next_training_time"] = datetime.now(
                    timezone.utc
                ) + timedelta(hours=RETRAIN_INTERVAL_HOURS)


def run_backtest():
    """Run backtest using the SAME logic as live trading (WITH FEES)"""
    print(f"\n{'='*60}")
    print(f"📊 RUNNING BACKTEST for {args.days} days")
    print(f"{'='*60}\n")

    # Fetch historical data
    df = fetch_ohlc(PAIR, INTERVAL, args.days)
    if df.empty:
        print("[backtest] No data available")
        return

    df = add_indicators(df)
    df.dropna(inplace=True)

    if df.empty:
        print("[backtest] Not enough data after indicators")
        return

    print(f"[backtest] Loaded {len(df)} candles for backtesting")

    # Initialize backtest state
    balance = 100.0  # Start with $100
    position = 0.0
    trades = []
    last_buy_price = 0
    highest_price_since_buy = 0

    # Track metrics
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0
    total_fees = 0

    print(f"\n{'='*60}")
    print(f"BACKTESTING {PAIR} on {INTERVAL}-minute candles")
    print(f"Strategy: AI + Adaptive RSI with Market Regime")
    print(f"Fees: {EXCHANGE_FEES['default'] * 100}% per trade (taker rate)")
    print(f"{'='*60}\n")

    for i in range(SEQUENCE_LENGTH, len(df)):
        current_row = df.iloc[i]
        price = current_row["close"]
        rsi = current_row[f"RSI_{INDICATOR_CONFIG['RSI']['length']}"]

        # Calculate trend for EMA50 filter
        ema_50 = current_row["EMA_50"]
        trend_up = price > ema_50
        trend_down = price < ema_50

        # Get ADX for trend strength
        adx = current_row["ADX_14"]
        strong_trend = adx > 25

        # Get MACD for display
        macd_line = current_row["MACD_12_26_9"]
        macd_signal = current_row["MACDs_12_26_9"]
        macd_bullish = macd_line > macd_signal

        # ============================================================
        # STRATEGY SELECTION BASED ON MARKET REGIME (CONSERVATIVE)
        # ============================================================
        if trend_up:
            buy_rsi_threshold = 30 if not strong_trend else 35
            sell_rsi_threshold = 70 if not strong_trend else 75
            strategy_name = "UPTREND"
        elif trend_down:
            buy_rsi_threshold = 15 if not strong_trend else 10
            sell_rsi_threshold = 50 if not strong_trend else 45
            strategy_name = "DOWNTREND"
        else:
            buy_rsi_threshold = 30
            sell_rsi_threshold = 70
            strategy_name = "SIDEWAYS"

        final_signal = "HOLD"

        # BUY signal
        if position == 0 and rsi <= buy_rsi_threshold:
            # Apply 1% price movement filter
            if len(trades) > 0:
                last_trade_price = trades[-1]["price"]
                price_change_pct = abs(price - last_trade_price) / last_trade_price
                if price_change_pct < 0.01:
                    continue

            # In downtrend with strong trend, require price reversal
            if trend_down and strong_trend:
                if i >= 2:
                    prev_low = df["low"].iloc[i - 1]
                    if price > prev_low:
                        final_signal = "BUY"
                    else:
                        final_signal = "HOLD"
                else:
                    final_signal = "HOLD"
            else:
                final_signal = "BUY"

        # Update trailing stop for existing position
        if position > 0:
            if price > highest_price_since_buy:
                highest_price_since_buy = price

            # Update trailing stop (0.5%)
            trailing_stop = highest_price_since_buy * 0.995

            # SELL signal
            sell_signal = (rsi >= sell_rsi_threshold and trend_down) or (
                price <= trailing_stop
            )

            if sell_signal and position > 0:
                final_signal = "SELL"

        # Execute BUY (WITH FEES)
        if final_signal == "BUY" and position == 0:
            # Calculate position size (95% of balance)
            position_size_usd = balance  # removed  * 0.95 after balance
            fee = position_size_usd * EXCHANGE_FEES["default"]
            position_size_usd_after_fee = position_size_usd - fee

            position = position_size_usd_after_fee / price
            balance = 0  # Keep 5% dust, was balance * 0.05
            total_fees += fee

            last_buy_price = price
            highest_price_since_buy = price

            trades.append(
                {
                    "time": df.index[i],
                    "type": "BUY",
                    "price": price,
                    "amount": position,
                    "cost": position * price,
                    "fee": fee,
                    "balance": balance,
                    "rsi": rsi,
                    "adx": adx,
                    "strategy": strategy_name,
                    "buy_threshold": buy_rsi_threshold,
                }
            )

            if args.verbose:
                print(
                    f"[{df.index[i].strftime('%Y-%m-%d %H:%M')}] 📈 BUY at ${price:.2f} (RSI: {rsi:.1f}) | Fee: ${fee:.4f} | Position: {position:.6f} BTC | Cash: ${balance:.2f}"
                )

        # Execute SELL (WITH FEES)
        elif final_signal == "SELL" and position > 0:
            proceeds_before_fee = position * price
            fee = proceeds_before_fee * EXCHANGE_FEES["default"]
            proceeds = proceeds_before_fee - fee

            pnl = proceeds - (position * last_buy_price)
            pnl_pct = (price - last_buy_price) / last_buy_price * 100

            balance += proceeds
            total_fees += fee

            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            total_pnl += pnl

            reason = (
                "RSI" if (rsi >= sell_rsi_threshold and trend_down) else "Trailing Stop"
            )

            trades.append(
                {
                    "time": df.index[i],
                    "type": "SELL",
                    "price": price,
                    "amount": position,
                    "proceeds": proceeds,
                    "fee": fee,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "balance": balance,
                    "reason": reason,
                    "rsi": rsi,
                    "adx": adx,
                    "strategy": strategy_name,
                    "sell_threshold": sell_rsi_threshold,
                }
            )

            if args.verbose:
                pnl_symbol = "+" if pnl >= 0 else ""
                print(
                    f"[{df.index[i].strftime('%Y-%m-%d %H:%M')}] 📉 SELL at ${price:.2f} ({reason}) | Fee: ${fee:.4f} | P&L: {pnl_symbol}${pnl:.2f} ({pnl_pct:+.2f}%) | Cash: ${balance:.2f}"
                )

            position = 0
            last_buy_price = 0
            highest_price_since_buy = 0

    # Close any open position at the end (WITH FEES)
    if position > 0:
        final_price = df.iloc[-1]["close"]
        proceeds_before_fee = position * final_price
        fee = proceeds_before_fee * EXCHANGE_FEES["default"]
        proceeds = proceeds_before_fee - fee
        pnl = proceeds - (position * last_buy_price)
        balance += proceeds
        total_fees += fee
        total_pnl += pnl

        trades.append(
            {
                "time": df.index[-1],
                "type": "CLOSE",
                "price": final_price,
                "amount": position,
                "proceeds": proceeds,
                "fee": fee,
                "pnl": pnl,
            }
        )
        position = 0

    # Calculate final metrics
    total_trades = len([t for t in trades if t["type"] in ["BUY", "SELL"]])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    final_value = balance
    total_return = (final_value - 100) / 100 * 100

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST RESULTS (with {EXCHANGE_FEES['default']*100}% fees)")
    print(f"{'='*60}")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Fees Paid: ${total_fees:.4f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Final Balance: ${final_value:.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"{'='*60}\n")

    # Show all trades if verbose
    if args.verbose:
        print("\n📋 All Trades:")
        print("-" * 80)
        for trade in trades:
            if trade["type"] == "BUY":
                print(
                    f"{trade['time'].strftime('%m/%d %H:%M')} | BUY  | ${trade['price']:.2f} | Fee: ${trade['fee']:.4f} | RSI: {trade['rsi']:.1f} | {trade['strategy']}"
                )
            elif trade["type"] == "SELL":
                print(
                    f"{trade['time'].strftime('%m/%d %H:%M')} | SELL | ${trade['price']:.2f} | Fee: ${trade['fee']:.4f} | P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) | {trade['reason']}"
                )

    return trades


def train_only():
    """Train the model and save it without trading"""
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING MODE - Training model on {args.days} days of data")
    print(f"{'='*60}\n")

    # Override training days with command line argument
    global LOOKBACK_DAYS_TRAINING
    original_days = LOOKBACK_DAYS_TRAINING
    LOOKBACK_DAYS_TRAINING = args.days

    # Train the model
    retrain_model()

    # Restore original setting
    LOOKBACK_DAYS_TRAINING = original_days

    print(f"\n✅ Training complete! Model saved to {BEST_MODEL_PATH}")
    print(f"   Scaler saved to {BEST_SCALER_PATH}")
    print(
        f"\n   To use this model in live trading, run: python {sys.argv[0]} --mode live"
    )


def log_trade_to_csv(trade_type, volume, price, txid, fees):
    """Log trade to CSV file"""
    import csv
    from datetime import datetime

    log_file = "trade_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Timestamp", "Type", "Volume", "Price", "Total", "Fees", "TXID"]
            )

        total = volume * price
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trade_type.upper(),
                f"{volume:.8f}",
                f"{price:.8f}",
                f"{total:.2f}",
                f"{fees:.4f}",
                txid,
            ]
        )
    print(f"📝 Trade logged to {log_file}")


def execute_live_trade(trade_type, volume, price):
    """Execute real trade on Kraken"""
    try:
        print(
            f"🔥 EXECUTING LIVE {trade_type.upper()}: {volume:.8f} {CRYPTO_NAME} @ ~${price:.5f}"
        )

        order_response = kraken_ex.query_private(
            "AddOrder",
            {
                "pair": PAIR,
                "type": trade_type,
                "ordertype": "market",
                "volume": str(volume),
                "oflags": "fciq",
            },
        )

        if "error" in order_response and order_response["error"]:
            print(f"❌ Kraken Error: {order_response['error']}")
            return False, None

        txid = order_response["result"]["txid"][0]
        fees = volume * price * EXCHANGE_FEES["default"]

        print(f"✅ ORDER PLACED! TXID: {txid}")

        # Log to CSV
        log_trade_to_csv(trade_type, volume, price, txid, fees)

        # Add to app_state for Flask dashboard
        trade_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": trade_type.upper(),
            "amount": volume,
            "price": price,
            "cost": volume * price if trade_type == "buy" else None,
            "proceeds": volume * price if trade_type == "sell" else None,
            "fees": fees,
            "txid": txid[:12] + "...",
        }

        with app_state_lock:
            if "trade_history" not in app_state:
                app_state["trade_history"] = []
            app_state["trade_history"].append(trade_record)
            if len(app_state["trade_history"]) > 50:
                app_state["trade_history"] = app_state["trade_history"][-50:]

        # Send Discord notification
        send_trade_alert(trade_type, volume, price, txid)

        return True, txid
    except Exception as e:
        print(f"❌ Trade error: {e}")
        return False, None


def trading_loop():
    global trained_model, final_scaler

    print("[trading] Starting trading loop...")

    # Get initial balances
    crypto_balance, usdt_balance = get_live_balance()
    if crypto_balance is None:
        crypto_balance = 0
        usdt_balance = 0

    print(
        f"[trading] Initial balances: {CRYPTO_NAME}={crypto_balance:.4f}, USDT={usdt_balance:.2f}"
    )

    # Load existing model
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
        try:
            trained_model = build_model(len(FEATURES_LIST)).to(DEVICE)
            trained_model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            )
            trained_model.eval()
            final_scaler = joblib.load(BEST_SCALER_PATH)
            print("[trading] Loaded existing AI model")
        except Exception as e:
            print(f"[trading] Error loading model: {e}")
            trained_model = None

    # Training schedule
    next_train_time = datetime.now(timezone.utc)

    # Trading state
    last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    last_buy_price = 0
    highest_price_since_buy = 0
    initial_portfolio_value = usdt_balance + crypto_balance * get_current_price()
    last_status_time = datetime.now(timezone.utc) - timedelta(hours=1)

    # Minimum price movement tracking
    last_trade_price = 0
    MIN_PRICE_MOVEMENT_PCT = 0.015  # 1% minimum movement required

    # Cache for OHLC data to avoid rate limiting
    last_fetch_time = None
    cached_df = None
    CACHE_DURATION_SECONDS = 120  # Fetch new data every 2 minutes

    # POSITION AGE TRACKING
    position_open_time = None
    last_position_alert_time = None
    POSITION_ALERT_HOURS = [24, 48, 72]

    # INITIALIZE APP STATE (ONCE, BEFORE THE LOOP)
    with app_state_lock:
        app_state["is_training"] = False
        app_state["training_status"] = "Ready"
        app_state["optimization_phase"] = "Idle"
        app_state["training_progress"] = 0
        app_state["crypto_balance"] = crypto_balance
        app_state["usdt_balance"] = usdt_balance
        app_state["portfolio_value"] = initial_portfolio_value
        app_state["pnl"] = 0
        app_state["pnl_percent"] = 0
        app_state["position_age_hours"] = 0
        app_state["position_open_time"] = None
        app_state["position_buy_price"] = 0
        app_state["position_unrealized_pnl"] = 0
        app_state["current_price"] = 0

    while True:
        try:
            now = datetime.now(timezone.utc)

            # POSITION AGE NOTIFICATION AND APP STATE UPDATE
            if position_open_time is not None:
                position_age_hours = (now - position_open_time).total_seconds() / 3600

                with app_state_lock:
                    app_state["position_age_hours"] = position_age_hours
                    app_state["position_open_time"] = position_open_time
                    app_state["position_buy_price"] = last_buy_price
                    app_state["current_price"] = price
                    if crypto_balance > 0 and last_buy_price > 0:
                        app_state["position_unrealized_pnl"] = (
                            price - last_buy_price
                        ) * crypto_balance
                    else:
                        app_state["position_unrealized_pnl"] = 0

                for alert_hour in POSITION_ALERT_HOURS:
                    if position_age_hours >= alert_hour:
                        alert_key = f"alert_{alert_hour}"
                        if last_position_alert_time is None or alert_key not in str(
                            last_position_alert_time
                        ):
                            print(f"⚠️⚠️⚠️ POSITION AGE ALERT ⚠️⚠️⚠️")
                            print(
                                f"Position has been open for {position_age_hours:.1f} hours ({alert_hour}+ hours)"
                            )
                            print(f"Current price: ${price:.2f}")
                            print(f"Buy price: ${last_buy_price:.2f}")
                            print(
                                f"Current P&L: ${(price - last_buy_price) * crypto_balance:.2f}"
                            )
                            if DISCORD_WEBHOOK_ENABLED:
                                send_discord_message(
                                    f"⚠️ **POSITION AGE ALERT**\n\n"
                                    f"Position open for {position_age_hours:.1f} hours\n"
                                    f"Buy Price: ${last_buy_price:.2f}\n"
                                    f"Current Price: ${price:.2f}\n"
                                    f"Unrealized P&L: ${(price - last_buy_price) * crypto_balance:.2f}\n"
                                    f"Consider manual review if needed.",
                                    "Position Alert",
                                )
                            if last_position_alert_time is None:
                                last_position_alert_time = {}
                            last_position_alert_time[alert_key] = now
                            break
            else:
                with app_state_lock:
                    app_state["position_age_hours"] = 0
                    app_state["position_open_time"] = None
                    app_state["position_buy_price"] = 0
                    app_state["position_unrealized_pnl"] = 0

            # Retrain if needed
            if trained_model is None or now >= next_train_time:
                retrain_model()
                next_train_time = now + timedelta(hours=RETRAIN_INTERVAL_HOURS)
                if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
                    try:
                        trained_model = build_model(len(FEATURES_LIST)).to(DEVICE)
                        trained_model.load_state_dict(
                            torch.load(BEST_MODEL_PATH, map_location=DEVICE)
                        )
                        trained_model.eval()
                        final_scaler = joblib.load(BEST_SCALER_PATH)
                        print("[trading] AI model reloaded after training")
                    except Exception as e:
                        print(f"[trading] Error reloading model: {e}")

            # Get current price
            price = get_current_price()
            if price <= 0:
                time.sleep(10)
                continue

            # CHECK MINIMUM PRICE MOVEMENT
            if last_trade_price > 0:
                price_change_pct = abs(price - last_trade_price) / last_trade_price
                if price_change_pct < MIN_PRICE_MOVEMENT_PCT:
                    print(
                        f"⏸️ Price movement {price_change_pct*100:.3f}% below {MIN_PRICE_MOVEMENT_PCT*100}% minimum. Waiting..."
                    )
                    time.sleep(DECISION_INTERVAL_SECONDS)
                    continue

            # Update portfolio value
            current_value = usdt_balance + crypto_balance * price
            pnl = current_value - initial_portfolio_value
            pnl_pct = (
                (pnl / initial_portfolio_value * 100)
                if initial_portfolio_value > 0
                else 0
            )

            # UPDATE APP STATE (DYNAMIC VALUES ONLY)
            with app_state_lock:
                app_state["crypto_balance"] = crypto_balance
                app_state["usdt_balance"] = usdt_balance
                app_state["portfolio_value"] = current_value
                app_state["pnl"] = pnl
                app_state["pnl_percent"] = pnl_pct
                app_state["last_decision_time"] = now
                app_state["current_price"] = price

            # Check cooldown
            if (now - last_trade_time).total_seconds() < 60:
                time.sleep(10)
                continue

            # FETCH DATA WITH CACHING
            if (
                last_fetch_time is None
                or (now - last_fetch_time).total_seconds() > CACHE_DURATION_SECONDS
            ):
                df = fetch_ohlc(PAIR, INTERVAL, 2)
                if not df.empty:
                    df = add_indicators(df)
                    cached_df = df
                    last_fetch_time = now
                    print(f"[cache] Fetched fresh data")
                else:
                    df = cached_df if cached_df is not None else pd.DataFrame()
            else:
                df = cached_df if cached_df is not None else pd.DataFrame()
                print(
                    f"[cache] Using cached data (age: {(now - last_fetch_time).total_seconds():.0f}s)"
                )

            if df.empty:
                time.sleep(10)
                continue

            rsi_col = f"RSI_{INDICATOR_CONFIG['RSI']['length']}"
            if rsi_col not in df.columns:
                print(
                    f"[warning] {rsi_col} not found in dataframe. Available columns: {list(df.columns)[:10]}..."
                )
                time.sleep(DECISION_INTERVAL_SECONDS)
                continue
            rsi = df[rsi_col].iloc[-1]

            # AI MODEL PREDICTION
            ai_signal = "HOLD"
            ai_confidence = 0.0
            ai_predicted_change = 0.0

            if trained_model is not None and final_scaler is not None:
                try:
                    features_tensor = prepare_ai_features(
                        df, final_scaler, SEQUENCE_LENGTH
                    )
                    if features_tensor is not None:
                        with torch.no_grad():
                            ai_prediction = trained_model(features_tensor)
                            if ai_prediction.numel() == 1:
                                ai_predicted_change = ai_prediction.item()
                            else:
                                ai_predicted_change = ai_prediction.mean().item()

                            if ai_predicted_change > 0.001:
                                ai_signal = "BUY"
                                ai_confidence = min(100, ai_predicted_change * 100)
                            elif ai_predicted_change < -0.001:
                                ai_signal = "SELL"
                                ai_confidence = min(100, abs(ai_predicted_change) * 100)
                            else:
                                ai_signal = "HOLD"

                            print(
                                f"🤖 AI PREDICTION: {ai_predicted_change:+.4f}% | Signal: {ai_signal} | Confidence: {ai_confidence:.1f}%"
                            )
                except Exception as e:
                    print(f"⚠️ AI prediction error: {e}")
            else:
                print("⚠️ No AI model available - using RSI-only fallback")

            # EMA50 TREND FILTER
            ema_50 = df["EMA_50"].iloc[-1]
            trend_up = price > ema_50
            trend_down = price < ema_50

            # ADX TREND STRENGTH
            adx = df["ADX_14"].iloc[-1]
            strong_trend = adx > 25

            # MACD for display
            macd_line = df["MACD_12_26_9"].iloc[-1]
            macd_signal_line = df["MACDs_12_26_9"].iloc[-1]
            macd_bullish = macd_line > macd_signal_line

            # STRATEGY SELECTION BASED ON MARKET REGIME
            if trend_up:
                buy_rsi_threshold = 35
                sell_rsi_threshold = 75
                strategy_name = "UPTREND - Buy dips, sell rips"
                if strong_trend:
                    buy_rsi_threshold = 40
                    sell_rsi_threshold = 80
                    strategy_name = "STRONG UPTREND - Aggressive buys, patient sells"
            elif trend_down:
                buy_rsi_threshold = 20
                sell_rsi_threshold = 60
                strategy_name = "DOWNTREND - Sell rips, buy only extreme dips"
                if strong_trend:
                    buy_rsi_threshold = 10
                    sell_rsi_threshold = 55
                    strategy_name = "STRONG DOWNTREND - Avoid buys, sell strength"
            else:
                buy_rsi_threshold = 30
                sell_rsi_threshold = 70
                strategy_name = "SIDEWAYS - Range trading"

            print(f"📊 STRATEGY: {strategy_name}")
            print(
                f"📊 RSI: {rsi:.1f} | ADX: {adx:.1f} {'(STRONG TREND)' if strong_trend else '(WEAK TREND)'}"
            )
            print(
                f"📊 Buy Threshold: RSI ≤ {buy_rsi_threshold} | Sell Threshold: RSI ≥ {sell_rsi_threshold}"
            )
            print(f"📊 MACD: {'BULLISH' if macd_bullish else 'BEARISH'}")
            print(f"🤖 AI: {ai_signal} ({ai_confidence:.0f}%) | Price: ${price:.2f}")
            print(f"📊 {CRYPTO_NAME}: {crypto_balance:.4f} | USDT: ${usdt_balance:.2f}")

            # Send status update every hour
            if (now - last_status_time).total_seconds() > 3600:
                send_status_update(
                    crypto_balance,
                    usdt_balance,
                    current_value,
                    rsi,
                    price,
                    pnl,
                    pnl_pct,
                )
                last_status_time = now

            # FINAL DECISION LOGIC
            final_signal = ai_signal

            # RSI OVERRIDE WITH MARKET REGIME AWARENESS
            if rsi <= buy_rsi_threshold and usdt_balance >= MIN_TRADE_AMOUNT:
                if trend_down and strong_trend:
                    if len(df) >= 3:
                        prev_low = df["low"].iloc[-2]
                        if price > prev_low:
                            print(
                                f"✅ DOWNTREND BUY: RSI {rsi:.1f} ≤ {buy_rsi_threshold} + price reversal confirmed"
                            )
                            final_signal = "BUY"
                        else:
                            print(
                                f"🚫 DOWNTREND BUY BLOCKED: RSI {rsi:.1f} ≤ {buy_rsi_threshold} but no reversal yet"
                            )
                            final_signal = "HOLD"
                    else:
                        final_signal = "HOLD"
                else:
                    print(f"📈 RSI TRIGGER: RSI {rsi:.1f} ≤ {buy_rsi_threshold} -> BUY")
                    final_signal = "BUY"

            if rsi >= sell_rsi_threshold and crypto_balance > 0:
                if trend_up and strong_trend and rsi < 85:
                    print(
                        f"📈 UPTREND SELL: RSI {rsi:.1f} ≥ {sell_rsi_threshold} but strong trend - holding longer"
                    )
                    final_signal = "HOLD"
                else:
                    print(
                        f"📉 RSI TRIGGER: RSI {rsi:.1f} ≥ {sell_rsi_threshold} -> SELL"
                    )
                    final_signal = "SELL"

            # EXECUTE SELL
            if final_signal == "SELL" and crypto_balance > 0:
                volume = crypto_balance * TRADE_PERCENTAGE

                if CRYPTO_NAME == "BTC" and volume < 0.0001:
                    print(
                        f"⚠️ Volume {volume:.8f} BTC below Kraken minimum (0.0001 BTC). Cannot sell."
                    )
                    time.sleep(DECISION_INTERVAL_SECONDS)
                    continue
                elif CRYPTO_NAME == "DOGE" and volume < 15.0:
                    print(
                        f"⚠️ Volume {volume:.2f} DOGE below Kraken minimum (15 DOGE). Cannot sell."
                    )
                    time.sleep(DECISION_INTERVAL_SECONDS)
                    continue

                if volume * price >= MIN_TRADE_AMOUNT:
                    print(f"🔥 SELL SIGNAL! (Confidence: {ai_confidence:.1f}%)")
                    if LIVE_TRADING:
                        success, txid = execute_live_trade("sell", volume, price)
                        if success:
                            proceeds = volume * price * (1 - EXCHANGE_FEES["default"])
                            usdt_balance += proceeds
                            crypto_balance -= volume
                            last_trade_time = now
                            last_trade_price = price
                            last_buy_price = 0
                            position_open_time = None
                            last_position_alert_time = None
                            with app_state_lock:
                                app_state["position_age_hours"] = 0
                                app_state["position_open_time"] = None
                                app_state["position_buy_price"] = 0
                                app_state["position_unrealized_pnl"] = 0
                            print(f"✅ SOLD {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}")
                    else:
                        proceeds = volume * price * (1 - EXCHANGE_FEES["default"])
                        usdt_balance += proceeds
                        crypto_balance -= volume
                        last_trade_time = now
                        last_trade_price = price
                        last_buy_price = 0
                        position_open_time = None
                        last_position_alert_time = None
                        with app_state_lock:
                            app_state["position_age_hours"] = 0
                            app_state["position_open_time"] = None
                            app_state["position_buy_price"] = 0
                            app_state["position_unrealized_pnl"] = 0
                        print(
                            f"📝 PAPER SELL {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}"
                        )

            # EXECUTE BUY
            elif final_signal == "BUY" and usdt_balance >= MIN_TRADE_AMOUNT:
                max_buy_usdt = min(
                    usdt_balance * TRADE_PERCENTAGE, MAX_NOTIONAL_PER_TRADE
                )
                volume = max_buy_usdt / price

                if CRYPTO_NAME == "BTC" and volume < 0.0001:
                    print(
                        f"⚠️ Volume {volume:.8f} BTC below Kraken minimum (0.0001 BTC). Need more USDT to buy."
                    )
                    time.sleep(DECISION_INTERVAL_SECONDS)
                    continue

                if volume * price >= MIN_TRADE_AMOUNT:
                    print(f"📈 BUY SIGNAL! (Confidence: {ai_confidence:.1f}%)")
                    if LIVE_TRADING:
                        success, txid = execute_live_trade("buy", volume, price)
                        if success:
                            cost = volume * price * (1 + EXCHANGE_FEES["default"])
                            usdt_balance -= cost
                            crypto_balance += volume
                            last_trade_time = now
                            last_trade_price = price
                            last_buy_price = price
                            highest_price_since_buy = price
                            position_open_time = now
                            last_position_alert_time = None
                            print(
                                f"✅ BOUGHT {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}"
                            )
                    else:
                        cost = volume * price * (1 + EXCHANGE_FEES["default"])
                        usdt_balance -= cost
                        crypto_balance += volume
                        last_trade_time = now
                        last_trade_price = price
                        last_buy_price = price
                        highest_price_since_buy = price
                        position_open_time = now
                        last_position_alert_time = None
                        print(f"📝 PAPER BUY {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}")

            # TRAILING STOP
            if last_buy_price > 0 and price > highest_price_since_buy:
                highest_price_since_buy = price

            if last_buy_price > 0 and highest_price_since_buy > 0:
                drop_pct = (highest_price_since_buy - price) / highest_price_since_buy
                if drop_pct >= 0.01 and crypto_balance > 0:
                    volume = crypto_balance
                    print(f"🔴 TRAILING STOP! Dropped {drop_pct*100:.2f}% from peak")
                    if LIVE_TRADING:
                        success, txid = execute_live_trade("sell", volume, price)
                        if success:
                            proceeds = volume * price * (1 - EXCHANGE_FEES["default"])
                            usdt_balance += proceeds
                            crypto_balance -= volume
                            last_trade_time = now
                            last_trade_price = price
                            last_buy_price = 0
                            highest_price_since_buy = 0
                            position_open_time = None
                            last_position_alert_time = None
                            with app_state_lock:
                                app_state["position_age_hours"] = 0
                                app_state["position_open_time"] = None
                                app_state["position_buy_price"] = 0
                                app_state["position_unrealized_pnl"] = 0
                            print(
                                f"✅ TRAILING STOP SOLD {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}"
                            )
                    else:
                        proceeds = volume * price * (1 - EXCHANGE_FEES["default"])
                        usdt_balance += proceeds
                        crypto_balance -= volume
                        last_trade_time = now
                        last_trade_price = price
                        last_buy_price = 0
                        highest_price_since_buy = 0
                        position_open_time = None
                        last_position_alert_time = None
                        with app_state_lock:
                            app_state["position_age_hours"] = 0
                            app_state["position_open_time"] = None
                            app_state["position_buy_price"] = 0
                            app_state["position_unrealized_pnl"] = 0
                        print(
                            f"📝 TRAILING STOP SELL {volume:.4f} {CRYPTO_NAME} @ ${price:.5f}"
                        )

            with app_state_lock:
                app_state["current_price"] = price

            time.sleep(DECISION_INTERVAL_SECONDS)

        except Exception as e:
            print(f"[loop] Error: {e}")
            import traceback

            traceback.print_exc()
            time.sleep(30)


def format_local_display(utc_dt, format_str="%Y-%m-%d %I:%M:%S %p"):
    if utc_dt is None:
        return "Never"
    try:
        local_tz = get_localzone()
        if utc_dt.tzinfo is None:
            utc_dt = pytz.UTC.localize(utc_dt)
        local_dt = utc_dt.astimezone(local_tz)
        return local_dt.strftime(format_str)
    except Exception:
        if utc_dt.tzinfo is None:
            return utc_dt.strftime(format_str) + " (UTC)"
        return utc_dt.astimezone(pytz.UTC).strftime(format_str) + " (UTC)"


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Advanced Crypto Trading Bot</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #0f0f23; color: #00ff00; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: #1a1a2e; padding: 20px; margin: 10px 0; border-radius: 8px; border: 1px solid #00ff00; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .status-badge { padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
        .status-training { background: #ffa500; }
        .status-ready { background: #28a745; }
        .status-live { background: #dc3545; }
        .status-paper { background: #17a2b8; }
        .progress-bar { width: 100%; background: #2d2d2d; border-radius: 4px; margin: 10px 0; }
        .progress-fill { height: 20px; background: linear-gradient(90deg, #00ff00, #00cc00); border-radius: 4px; text-align: center; color: white; line-height: 20px; }
        .trade-history { max-height: 300px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #00ff00; }
        th { background: #2a2a3c; }
        .buy { color: #00ff00; font-weight: bold; }
        .sell { color: #ff4444; font-weight: bold; }
        .hold { color: #ffff00; }
        .warning { background: #ff4444; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .optimization-info { background: #2a2a3c; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .fee-info { background: #2a3c2a; padding: 10px; border-radius: 4px; margin: 5px 0; font-size: 0.9em; }
        .target-price-box { margin-top: 15px; padding: 10px; background: #2a2a3c; border-radius: 4px; }
        .reasoning-box { background: #2a2a3c; padding: 10px; border-radius: 4px; margin-top: 10px; font-size: 0.9em; color: #cccccc; }
        .position-age-box { margin-top: 15px; padding: 10px; background: #2a3c2a; border-radius: 4px; border-left: 3px solid #ffaa00; }
        .position-age-critical { background: #3c2a2a; border-left: 3px solid #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Advanced AI Crypto Trading Bot</h1>

        {% if LIVE_TRADING %}
        <div class="warning">
            ⚠️ <strong>LIVE TRADING MODE</strong> - Real money is at risk!
        </div>
        {% endif %}

        <div class="grid">
            <!-- Trading Status -->
            <div class="card">
                <h2>📊 Trading Status</h2>
                <p><strong>Mode:</strong> <span class="status-badge {% if LIVE_TRADING %}status-live{% else %}status-paper{% endif %}">
                    {{ "LIVE TRADING" if LIVE_TRADING else "PAPER TRADING" }}
                </span></p>
                <p><strong>Pair:</strong> {{ trading_pair }}</p>
                <p><strong>Device:</strong> {{ device }}</p>
                <p><strong>Strategy:</strong> <span style="color: #ff4444;">🔥 AGGRESSIVE</span></p>
                <p><strong>Trade Size:</strong> {{ (TRADE_PERCENTAGE * 100)|int }}% of balance</p>
                <p><strong>Fees:</strong> {{ (EXCHANGE_FEES.default * 100)|float }}% per trade</p>
                <p><strong>Last Decision:</strong> {{ last_decision_time }}</p>
                
                <!-- Target Price Display -->
                {% if target_price_info and target_price_info.last_buy_price > 0 %}
                <div class="target-price-box">
                    <h3 style="margin-top: 0; color: #00ff00;">🎯 Target Price</h3>
                    <p><strong>Last Buy:</strong> ${{ "%.2f"|format(target_price_info.last_buy_price) }}</p>
                    <p><strong>Current Price:</strong> ${{ "%.2f"|format(target_price_info.current_price) }}</p>
                    <p><strong>Target Sell Price:</strong> 
                        <span style="color: {% if target_price_info.current_price >= target_price_info.target_sell_price %}#00ff00{% else %}#ffff00{% endif %}; font-weight: bold;">
                            ${{ "%.2f"|format(target_price_info.target_sell_price) }}
                        </span>
                    </p>
                    <p><strong>Need Price to Reach:</strong> 
                        <span style="color: {% if target_price_info.price_to_go <= 0 %}#00ff00{% else %}#ff4444{% endif %}">
                            +${{ "%.2f"|format(target_price_info.price_to_go) }} 
                            ({{ "%.2f"|format(target_price_info.pct_to_go) }}%)
                        </span>
                    </p>
                    <p><strong>Minimum Profit:</strong> {{ "%.2f"|format(target_price_info.needed_profit_pct) }}% after fees</p>
                    
                    {% if target_price_info.current_price >= target_price_info.target_sell_price %}
                    <p style="color: #00ff00; font-weight: bold;">✅ READY TO SELL!</p>
                    {% else %}
                    <div class="progress-bar" style="margin-top: 10px;">
                        {% set progress_percent = [100, (target_price_info.current_price / target_price_info.target_sell_price * 100)|int]|min %}
                        <div class="progress-fill" style="width: {{ progress_percent }}%;">
                            {{ progress_percent }}%
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% elif target_price_info %}
                <div class="target-price-box">
                    <h3 style="margin-top: 0; color: #00ff00;">🎯 Target Price</h3>
                    <p>Waiting for buy signal...</p>
                </div>
                {% endif %}
                
                <!-- Reasoning AI Analysis -->
                {% if reasoning_analysis and reasoning_analysis.reasoning %}
                <div class="reasoning-box">
                    <h4 style="margin: 0 0 5px 0;">🧠 AI Reasoning</h4>
                    <p><strong>Signal:</strong> {{ reasoning_analysis.recommended_action|upper }}</p>
                    <p><strong>Reasoning:</strong> {{ reasoning_analysis.reasoning[:200] }}</p>
                </div>
                {% endif %}
            </div>

            <!-- Balances -->
            <div class="card">
                <h2>💰 Balances & Performance</h2>
                <p><strong>{{ crypto_name }} Balance:</strong> {{ "%.4f"|format(crypto_balance) }}</p>
                <p><strong>USDT Balance:</strong> {{ "%.2f"|format(usdt_balance) }}</p>
                <p><strong>Portfolio Value:</strong> {{ "%.2f"|format(portfolio_value) }} USDT</p>
                <p><strong>Profit/Loss:</strong> <span style="color: {{ 'green' if pnl >= 0 else 'red' }}; font-weight: bold;">
                    {{ "%.2f"|format(pnl) }} USDT ({{ "%.2f"|format(pnl_percent) }}%)
                </span></p>
                
                <!-- Position Age Display -->
                {% if position_age_hours is defined and position_age_hours > 0 %}
                <div class="position-age-box {% if position_age_hours >= 48 %}position-age-critical{% endif %}">
                    <h4 style="margin: 0 0 8px 0;">⏰ Position Age</h4>
                    <p><strong>Open Since:</strong> {{ position_open_time }}</p>
                    <p><strong>Age:</strong> 
                        <span style="color: {% if position_age_hours >= 48 %}#ff4444{% elif position_age_hours >= 24 %}#ffaa00{% else %}#00ff00{% endif %}; font-weight: bold;">
                            {{ "%.1f"|format(position_age_hours) }} hours
                        </span>
                    </p>
                    <p><strong>Buy Price:</strong> ${{ "%.2f"|format(position_buy_price) }}</p>
                    <p><strong>Current Price:</strong> ${{ "%.2f"|format(current_price) }}</p>
                    <p><strong>Unrealized P&L:</strong> 
                        <span style="color: {{ 'green' if position_unrealized_pnl >= 0 else 'red' }};">
                            ${{ "%.2f"|format(position_unrealized_pnl) }}
                        </span>
                    </p>
                    {% if position_age_hours >= 48 %}
                    <p style="color: #ff4444; margin-top: 8px;">⚠️ Position has been open for over 48 hours - consider manual review!</p>
                    {% elif position_age_hours >= 24 %}
                    <p style="color: #ffaa00; margin-top: 8px;">⚠️ Position open for over 24 hours - monitoring...</p>
                    {% endif %}
                </div>
                {% elif crypto_balance > 0 %}
                <div class="position-age-box">
                    <h4 style="margin: 0 0 8px 0;">⏰ Position Age</h4>
                    <p>Position tracking active - age data loading...</p>
                </div>
                {% endif %}
                
                <div class="fee-info">
                    <strong>Fee-Aware Trading:</strong> All calculations include {{ (EXCHANGE_FEES.default * 100)|float }}% trading fees
                </div>
            </div>

            <!-- Optimization Results -->
            <div class="card">
                <h2>🎯 Optimization Results</h2>
                <p><strong>Best ATR Multiplier:</strong> {{ best_multiplier }}</p>
                <p><strong>Best Learning Rate:</strong> {{ best_learning_rate }}</p>
                <p><strong>Last Training:</strong> {{ last_training_time }}</p>
                <p><strong>Next Training:</strong> {{ next_training_time }}</p>
                <p><strong>Optimization Phase:</strong> {{ optimization_phase }}</p>
            </div>
        </div>

        <!-- Training Progress -->
        <div class="card">
            <h2>🧠 AI Training Progress</h2>
            <p><strong>Status:</strong> <span class="status-badge {% if is_training %}status-training{% else %}status-ready{% endif %}">
                {{ training_status }}
            </span></p>
            {% if is_training %}
            <div class="optimization-info">
                <p><strong>ATR Multiplier:</strong> {{ current_multiplier }}</p>
                <p><strong>Learning Rate:</strong> {{ current_learning_rate }}</p>
                <p><strong>Run:</strong> {{ current_run }}/{{ total_runs }}</p>
                <p><strong>Epoch:</strong> {{ current_epoch }}/{{ total_epochs }}</p>
                <p><strong>Loss:</strong> {{ "%.6f"|format(current_loss) }}</p>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ training_progress }}%;">{{ training_progress }}%</div>
            </div>
            {% endif %}
        </div>

        <!-- Recent Trades -->
        <div class="card">
            <h2>📈 Recent Trades (Aggressive Mode)</h2>
            <div class="trade-history">
                {% if trade_history %}
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>Amount</th>
                            <th>Price</th>
                            <th>Total</th>
                            <th>Fees</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in trade_history[-15:] %}
                        <tr>
                            <td>{{ trade.time }}</td>
                            <td class="{{ trade.type.lower() }}">{{ trade.type }}</td>
                            <td>{{ "%.4f"|format(trade.amount) }}</td>
                            <td>{{ "%.5f"|format(trade.price) }}</td>
                            <td>{{ "%.2f"|format(trade.cost if trade.type == 'BUY' else trade.proceeds) }}</td>
                            <td>{{ "%.4f"|format(trade.fees) }} USDT</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <p>No trades yet</p>
                {% endif %}
            </div>
        </div>

        <!-- Trading Strategy Info -->
        <div class="card">
            <h2>⚡ Trading Strategy</h2>
            <p><strong>Aggressive Settings:</strong></p>
            <ul>
                <li>Trade Size: {{ (TRADE_PERCENTAGE * 100)|int }}% per trade</li>
                <li>Trading Fees: {{ (EXCHANGE_FEES.default * 100)|float }}% per trade</li>
                <li>RSI Overbought: {{ INDICATOR_CONFIG.RSI.overbought }} (Tight)</li>
                <li>RSI Oversold: {{ INDICATOR_CONFIG.RSI.oversold }} (Tight)</li>
                <li>Fast MACD: {{ INDICATOR_CONFIG.MACD.fast }}/{{ INDICATOR_CONFIG.MACD.slow }}/{{ INDICATOR_CONFIG.MACD.signal }}</li>
                <li>ATR Multipliers Tested: {{ ATR_MULTIPLIERS_TO_TEST|length }}</li>
                <li>Learning Rates Tested: {{ LEARNING_RATES_TO_TEST|length }}</li>
                <li>Minimum Profit Required: 0.5% after fees</li>
                <li>Trade Cooldown: 1 minute between trades</li>
            </ul>
        </div>
    </div>

    <script>
        // Auto-refresh more frequently during training
        {% if is_training %}
        setTimeout(() => location.reload(), 3000);
        {% else %}
        setTimeout(() => location.reload(), 10000);
        {% endif %}
    </script>
</body>
</html>"""


@app.route("/")
def dashboard():
    with app_state_lock:
        position_age_hours = app_state.get("position_age_hours", 0)
        position_open_time = app_state.get("position_open_time", None)
        position_buy_price = app_state.get("position_buy_price", 0)
        position_unrealized_pnl = app_state.get("position_unrealized_pnl", 0)
        current_price = app_state.get("current_price", 0)

        return render_template_string(
            HTML_TEMPLATE,
            LIVE_TRADING=LIVE_TRADING,
            trading_pair=PAIR,
            device=str(DEVICE),
            TRADE_PERCENTAGE=TRADE_PERCENTAGE,
            EXCHANGE_FEES=EXCHANGE_FEES,
            crypto_name=CRYPTO_NAME,
            crypto_balance=app_state.get("crypto_balance", 0),
            usdt_balance=app_state.get("usdt_balance", 0),
            portfolio_value=app_state.get("portfolio_value", 0),
            pnl=app_state.get("pnl", 0),
            pnl_percent=app_state.get("pnl_percent", 0),
            best_multiplier=app_state.get("best_multiplier", "N/A"),
            best_learning_rate=app_state.get("best_learning_rate", "N/A"),
            last_training_time=format_local_display(
                app_state.get("last_training_time")
            ),
            next_training_time=format_local_display(
                app_state.get("next_training_time")
            ),
            optimization_phase=app_state.get("optimization_phase", "Not Started"),
            training_status=app_state.get("training_status", "Not Training"),
            is_training=app_state.get("is_training", False),
            current_multiplier=app_state.get("current_multiplier", "N/A"),
            current_learning_rate=app_state.get("current_learning_rate", "N/A"),
            current_run=app_state.get("current_run", 0),
            total_runs=app_state.get("total_runs", 0),
            current_epoch=app_state.get("current_epoch", 0),
            total_epochs=app_state.get("total_epochs", 0),
            current_loss=app_state.get("current_loss", 0.0),
            training_progress=app_state.get("training_progress", 0),
            last_decision_time=format_local_display(
                app_state.get("last_decision_time")
            ),
            trade_history=app_state.get("trade_history", []),
            target_price_info=app_state.get("target_price_info", None),
            reasoning_analysis=app_state.get("claude_analysis", {}),
            INDICATOR_CONFIG=INDICATOR_CONFIG,
            ATR_MULTIPLIERS_TO_TEST=ATR_MULTIPLIERS_TO_TEST,
            LEARNING_RATES_TO_TEST=LEARNING_RATES_TO_TEST,
            position_age_hours=position_age_hours,
            position_open_time=(
                format_local_display(position_open_time)
                if position_open_time
                else "No open position"
            ),
            position_buy_price=position_buy_price,
            position_unrealized_pnl=position_unrealized_pnl,
            current_price=current_price,
        )


def main():
    print(f"[main] Starting bot for {CRYPTO_NAME}...")

    # Handle different modes
    if args.mode == "backtest":
        print("📊 BACKTEST MODE - Simulating trades with historical data")
        run_backtest()
        return

    elif args.mode == "train":
        print("🧠 TRAIN MODE - Training model without trading")
        train_only()
        return

    elif args.mode == "live":
        print("🔴 LIVE MODE - Real trading with real money")

        # Test Discord connection on startup
        if DISCORD_WEBHOOK_ENABLED and DISCORD_WEBHOOK_URL:
            send_discord_message(
                f"🤖 Bot Started\n\nPair: {PAIR}\nMode: {'LIVE' if LIVE_TRADING else 'PAPER'}\nStatus: Online",
                "Bot Online",
            )

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        trading_loop()

        return


if __name__ == "__main__":
    main()
