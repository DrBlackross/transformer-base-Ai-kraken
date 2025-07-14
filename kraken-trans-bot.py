import pandas as pd
import pandas_ta as ta
import krakenex
from pykrakenapi import KrakenAPI
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime, timedelta, UTC
import os
import joblib
import shutil
import warnings
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# ======================
# USER CONFIGURABLE SETTINGS
# ======================

# --- Core Trading Configuration ---
LIVE_TRADING = False  # Set to True for real trading, False for paper trading
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')  # Your Kraken API key (set in environment variables)
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')  # Your Kraken API secret (set in environment variables)

# Select trading pair (DOGEUSDT or XBTUSDT)
TRADING_PAIR = 'XBTUSDT'  # Options: 'DOGEUSDT' (Dogecoin) or 'XBTUSDT' (Bitcoin)

# --- Trading Pair Specific Settings ---
if TRADING_PAIR == 'DOGEUSDT':
    MODEL_PATH_BASE = './doge_transformer_model'  # Path to save Dogecoin model files
    INITIAL_USDT_BALANCE = 10.0  # Starting USDT balance for Dogecoin trading
    INITIAL_CRYPTO_BALANCE = 100.0  # Starting DOGE balance
    MIN_TRADE_AMOUNT = 5.0  # Minimum trade size in USDT
    CRYPTO_NAME = 'DOGE'  # Display name
    CRYPTO_DECIMALS = 4  # Decimal precision for display
    DEFAULT_ATR_MULTIPLIER = 0.15  # Default volatility multiplier for Dogecoin
elif TRADING_PAIR == 'XBTUSDT':
    MODEL_PATH_BASE = './btc_transformer_model'  # Path to save Bitcoin model files
    INITIAL_USDT_BALANCE = 0.0  # Starting USDT balance for Bitcoin trading
    INITIAL_CRYPTO_BALANCE = 0.00149596  # Starting BTC balance (~$50 at $33,500/BTC)
    MIN_TRADE_AMOUNT = 10.0  # Minimum trade size in USDT
    CRYPTO_NAME = 'BTC'  # Display name
    CRYPTO_DECIMALS = 6  # Decimal precision for display
    DEFAULT_ATR_MULTIPLIER = 0.1  # Default volatility multiplier for Bitcoin
else:
    raise ValueError("Invalid trading pair. Use 'DOGEUSDT' or 'XBTUSDT'")

# --- Model Path Configuration ---
SCALER_PATH_BASE = './standard_scaler'  # Base path for feature scaler files
BEST_MODEL_PATH = f'{MODEL_PATH_BASE}_best.pth'  # Path for best model
BEST_SCALER_PATH = f'{SCALER_PATH_BASE}_best.pkl'  # Path for best scaler
BEST_MULTIPLIER_PATH = f'{MODEL_PATH_BASE}_best_multiplier.txt'  # Path for best ATR multiplier

# --- Trading Parameters ---
PAIR = TRADING_PAIR  # Trading pair symbol
INTERVAL = 1  # Candlestick interval in minutes (1, 5, 15, etc.)
LOOKBACK_DAYS_TRAINING = 730  # Days of historical data for training (~2 years)
LOOKBACK_DAYS_WINDOW = 2  # Days of data to keep in memory for live trading
SEQUENCE_LENGTH = 24  # Number of time steps in each training sequence (e.g., 24 = 24 minutes)

# --- Model Training Parameters ---
NUM_TRAINING_RUNS = 3  # Number of training runs per ATR multiplier (reduce for faster testing)
LEARNING_RATE = 1e-5  # Learning rate for the transformer model
PER_DEVICE_BATCH_SIZE = 8  # Batch size for training
NUM_TRAIN_EPOCHS = 100  # Maximum training epochs (reduce for faster testing)
EVAL_STEPS = 20  # Evaluate model every N steps
LOGGING_STEPS = 10  # Log training progress every N steps
SAVE_STEPS = 20  # Save model checkpoint every N steps
SAVE_TOTAL_LIMIT = 2  # Maximum number of checkpoints to keep
WEIGHT_DECAY = 0.01  # L2 regularization strength
DROPOUT_RATE = 0.3  # Dropout rate for model regularization
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement after N evaluations
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement to reset patience counter

# --- Trading Strategy Parameters ---
TRADE_PERCENTAGE = 0.9  # Percentage of balance to trade (0.9 = 90%)
DECISION_INTERVAL_SECONDS = INTERVAL * 300  # Time between trading decisions (5 hours for 1m candles)
SLEEP_TIME_SECONDS = DECISION_INTERVAL_SECONDS + 1  # Sleep time between iterations

# --- ATR Multiplier Optimization ---
ATR_MULTIPLIERS_TO_TEST = [0.05, 0.1, 0.15, 0.2]  # Test these volatility multipliers:
#   0.05 - Very sensitive (trades on small moves) - High-frequency strategies
#   0.1 - Moderate sensitivity - Medium-term trading (default for BTC)
#   0.15 - Less sensitive - Volatile assets (default for DOGE)
#   0.2+ - Only trades on large swings - Position trading

# --- Technical Indicator Configuration ---
INDICATOR_CONFIG = {
    'BASE_FEATURES': ['open', 'high', 'low', 'close', 'volume'],  # Core price data
    'RSI': {'enabled': True, 'length': 3, 'overbought': 80, 'oversold': 30},  # Relative Strength Index
    'MACD': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},  # Moving Average Convergence Divergence
    'BBANDS': {'enabled': True, 'length': 5, 'std': 2.0},  # Bollinger Bands
    'OBV': {'enabled': True},  # On-Balance Volume
    'ADX': {'enabled': True, 'length': 14},  # Average Directional Index
    'CCI': {'enabled': True, 'length': 14, 'c': 0.015},  # Commodity Channel Index
    'ATR': {'enabled': True, 'length': 14},  # Average True Range (volatility)
    'VWAP': {'enabled': True},  # Volume Weighted Average Price
    'NATR': {'enabled': True, 'length': 14},  # Normalized ATR
    'TRIX': {'enabled': True, 'length': 4, 'signal': 9},  # Triple Exponential Average
    'STOCH': {'enabled': True, 'k': 3, 'd': 4, 'smooth_k': 3},  # Stochastic Oscillator
    'EMA': {'enabled': True, 'lengths': [10, 20]}  # Exponential Moving Averages
}

# --- Generate FEATURES_LIST ---
FEATURES_LIST = INDICATOR_CONFIG['BASE_FEATURES'].copy()

if INDICATOR_CONFIG['RSI']['enabled']:
    FEATURES_LIST.append(f"RSI_{INDICATOR_CONFIG['RSI']['length']}")

if INDICATOR_CONFIG['MACD']['enabled']:
    macd = INDICATOR_CONFIG['MACD']
    FEATURES_LIST.extend([
        f"MACD_{macd['fast']}_{macd['slow']}_{macd['signal']}",
        f"MACDh_{macd['fast']}_{macd['slow']}_{macd['signal']}",
        f"MACDs_{macd['fast']}_{macd['slow']}_{macd['signal']}"
    ])

if INDICATOR_CONFIG['BBANDS']['enabled']:
    bb = INDICATOR_CONFIG['BBANDS']
    FEATURES_LIST.extend([
        f"BBL_{bb['length']}_{bb['std']}",
        f"BBM_{bb['length']}_{bb['std']}",
        f"BBU_{bb['length']}_{bb['std']}"
    ])

if INDICATOR_CONFIG['OBV']['enabled']:
    FEATURES_LIST.append('OBV')

if INDICATOR_CONFIG['ADX']['enabled']:
    FEATURES_LIST.append(f"ADX_{INDICATOR_CONFIG['ADX']['length']}")

if INDICATOR_CONFIG['CCI']['enabled']:
    cci = INDICATOR_CONFIG['CCI']
    FEATURES_LIST.append(f"CCI_{cci['length']}_{cci['c']}")

if INDICATOR_CONFIG['ATR']['enabled']:
    FEATURES_LIST.append(f"ATRr_{INDICATOR_CONFIG['ATR']['length']}")

if INDICATOR_CONFIG['VWAP']['enabled']:
    FEATURES_LIST.append('VWAP_D')

if INDICATOR_CONFIG['NATR']['enabled']:
    FEATURES_LIST.append(f"NATR_{INDICATOR_CONFIG['ATR']['length']}")

if INDICATOR_CONFIG['TRIX']['enabled']:
    trix = INDICATOR_CONFIG['TRIX']
    FEATURES_LIST.extend([
        f"TRIX_{trix['length']}_{trix['signal']}",
        f"TRIXs_{trix['length']}_{trix['signal']}"
    ])

if INDICATOR_CONFIG['STOCH']['enabled']:
    stoch = INDICATOR_CONFIG['STOCH']
    FEATURES_LIST.extend([
        f"STOCHk_{stoch['k']}_{stoch['d']}_{stoch['smooth_k']}",
        f"STOCHd_{stoch['k']}_{stoch['d']}_{stoch['smooth_k']}"
    ])

if INDICATOR_CONFIG['EMA']['enabled']:
    for length in INDICATOR_CONFIG['EMA']['lengths']:
        FEATURES_LIST.append(f"EMA_{length}")

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- API Setup ---
if LIVE_TRADING:
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise ValueError("API keys must be set for live trading.")
    k = krakenex.API(key=KRAKEN_API_KEY, secret=KRAKEN_API_SECRET)
    print("Kraken API initialized for LIVE TRADING.")
else:
    k = krakenex.API()
    print("Kraken API initialized for PAPER TRADING.")
api = KrakenAPI(k)


# --- Model Definition ---
class CryptoTransformer(torch.nn.Module):
    def __init__(self, num_features, dropout_rate=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(num_features, 128)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = torch.nn.Linear(128 * SEQUENCE_LENGTH, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))

    def forward(self, x, labels=None):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        if loss is not None:
            return loss, logits
        return logits


# --- Dataset Definition ---
class TradingDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        if idx + self.seq_length >= len(self.features):
            raise IndexError("Index out of bounds for sequence length")
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return {"x": torch.FloatTensor(x), "labels": torch.tensor(y, dtype=torch.long)}


# --- Trading Engine ---
class CryptoTrader:
    def __init__(self, model, api, scaler, device, live_trading, initial_crypto_balance, initial_usdt_balance,
                 trading_writer=None):
        self.model = model
        self.api = api
        self.scaler = scaler
        self.device = device
        self.live_trading = live_trading
        self.crypto_balance = initial_crypto_balance
        self.usdt_balance = initial_usdt_balance
        self.last_trade_time = datetime.now(UTC) - timedelta(seconds=DECISION_INTERVAL_SECONDS)
        self.trade_percentage = TRADE_PERCENTAGE
        self.min_trade_amount = MIN_TRADE_AMOUNT
        self.trade_history = []
        self.trading_writer = trading_writer
        self.trade_step = 0
        self.initial_portfolio_value = initial_usdt_balance + (initial_crypto_balance * self._get_current_price())

        self.status_colors = {
            'BUY': '\033[92m',
            'SELL': '\033[91m',
            'HOLD': '\033[93m',
            'RESET': '\033[0m'
        }
        print(f"CryptoTrader initialized for {CRYPTO_NAME}. Live Trading: {self.live_trading}")
        print(
            f"Initial Balances: {CRYPTO_NAME}={self.crypto_balance:.{CRYPTO_DECIMALS}f}, USDT={self.usdt_balance:.2f}")

    def _get_current_price(self):
        try:
            df = self._fetch_latest_data()
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
            return 0
        except Exception as e:
            print(f"Error getting current price: {e}")
            return 0

    def _print_status(self, action, price, confidence=None):
        color = self.status_colors.get(action, '')
        action_str = f"{color}{action}{self.status_colors['RESET']}"
        price_str = f"{price:.5f}"
        confidence_str = f" (Confidence: {confidence:.2f}%)" if confidence is not None else ""
        print(f"\n{'-' * 50}")
        print(f"|  Action: {action_str:<10} | Price: {price_str:<10}{confidence_str}")
        print(f"{'-' * 50}")
        print(f"|  {CRYPTO_NAME} Balance: {self.crypto_balance:.{CRYPTO_DECIMALS}f}")
        print(f"|  USDT Balance: {self.usdt_balance:.2f}")
        current_value = self.crypto_balance * price + self.usdt_balance
        print(f"|  Portfolio Value: {current_value:.2f} USDT")
        print(f"{'-' * 50}\n")

    def _fetch_latest_data(self):
        try:
            since_time = int((datetime.now(UTC) - timedelta(days=LOOKBACK_DAYS_WINDOW)).timestamp())
            df = self.api.get_ohlc_data(PAIR, interval=INTERVAL, since=since_time)[0]
            df.index = pd.to_datetime(df.index, unit='s')
            df = df[~df.index.duplicated(keep='first')]
            new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
            df = df.reindex(new_index)
            df = df.ffill().infer_objects(copy=False)
            return df
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return None

    def _calculate_indicators(self, df):
        try:
            if INDICATOR_CONFIG['RSI']['enabled']:
                df[f"RSI_{INDICATOR_CONFIG['RSI']['length']}"] = ta.rsi(df['close'],
                                                                        length=INDICATOR_CONFIG['RSI']['length'])

            if INDICATOR_CONFIG['MACD']['enabled']:
                macd = ta.macd(df['close'], fast=INDICATOR_CONFIG['MACD']['fast'],
                               slow=INDICATOR_CONFIG['MACD']['slow'],
                               signal=INDICATOR_CONFIG['MACD']['signal'])
                df = pd.concat([df, macd], axis=1)

            if INDICATOR_CONFIG['BBANDS']['enabled']:
                bbands = ta.bbands(df['close'], length=INDICATOR_CONFIG['BBANDS']['length'],
                                   std=INDICATOR_CONFIG['BBANDS']['std'])
                df = pd.concat([df, bbands], axis=1)

            if INDICATOR_CONFIG['OBV']['enabled']:
                df['OBV'] = ta.obv(df['close'], df['volume'])

            if INDICATOR_CONFIG['ADX']['enabled']:
                adx = ta.adx(df['high'], df['low'], df['close'], length=INDICATOR_CONFIG['ADX']['length'])
                df[f"ADX_{INDICATOR_CONFIG['ADX']['length']}"] = adx[f"ADX_{INDICATOR_CONFIG['ADX']['length']}"]

            if INDICATOR_CONFIG['CCI']['enabled']:
                df[f"CCI_{INDICATOR_CONFIG['CCI']['length']}_{INDICATOR_CONFIG['CCI']['c']}"] = ta.cci(
                    df['high'], df['low'], df['close'],
                    length=INDICATOR_CONFIG['CCI']['length'],
                    c=INDICATOR_CONFIG['CCI']['c']
                )

            if INDICATOR_CONFIG['ATR']['enabled']:
                atr = ta.atr(df['high'], df['low'], df['close'], length=INDICATOR_CONFIG['ATR']['length'])
                df[f"ATRr_{INDICATOR_CONFIG['ATR']['length']}"] = atr

            if INDICATOR_CONFIG['VWAP']['enabled']:
                df['VWAP_D'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

            if INDICATOR_CONFIG['NATR']['enabled']:
                df[f"NATR_{INDICATOR_CONFIG['ATR']['length']}"] = ta.natr(df['high'], df['low'], df['close'],
                                                                          length=INDICATOR_CONFIG['ATR']['length'])

            if INDICATOR_CONFIG['TRIX']['enabled']:
                trix = ta.trix(df['close'], length=INDICATOR_CONFIG['TRIX']['length'],
                               signal=INDICATOR_CONFIG['TRIX']['signal'])
                df = pd.concat([df, trix], axis=1)

            if INDICATOR_CONFIG['STOCH']['enabled']:
                stoch = ta.stoch(df['high'], df['low'], df['close'],
                                 k=INDICATOR_CONFIG['STOCH']['k'],
                                 d=INDICATOR_CONFIG['STOCH']['d'],
                                 smooth_k=INDICATOR_CONFIG['STOCH']['smooth_k'])
                df = pd.concat([df, stoch], axis=1)

            if INDICATOR_CONFIG['EMA']['enabled']:
                for length in INDICATOR_CONFIG['EMA']['lengths']:
                    df[f"EMA_{length}"] = ta.ema(df['close'], length=length)

            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return pd.DataFrame()

    def _prepare_features(self, df):
        if len(df) < SEQUENCE_LENGTH:
            print(f"Not enough data for a sequence. Need {SEQUENCE_LENGTH}, got {len(df)}.")
            return None

        try:
            latest_data = df[FEATURES_LIST].tail(SEQUENCE_LENGTH).values
            scaled_features = self.scaler.transform(latest_data)
            features_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
            return features_tensor
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None

    def _execute_trade(self, trade_type, amount, price=None):
        if price is None or price <= 0:
            print("Invalid price for trade execution")
            return False

        trade_time = datetime.now(UTC)
        timestamp = trade_time.timestamp()

        try:
            if trade_type == 'buy':
                cost_usdt = amount * price
                if cost_usdt < self.min_trade_amount:
                    print(f"Trade amount too small: {cost_usdt:.2f} USDT (minimum: {self.min_trade_amount} USDT)")
                    return False

                if self.usdt_balance >= cost_usdt:
                    self.usdt_balance -= cost_usdt
                    self.crypto_balance += amount
                    trade_record = {
                        'time': trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'BUY',
                        'amount': amount,
                        'price': price,
                        'cost': cost_usdt
                    }
                    self.trade_history.append(trade_record)

                    if self.trading_writer:
                        self.trading_writer.add_scalar('Trades/Buy_Price', price, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Buy_Amount', amount, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Buy_Cost', cost_usdt, self.trade_step)

                    print(
                        f"BUY successful. New balances: {CRYPTO_NAME}={self.crypto_balance:.{CRYPTO_DECIMALS}f}, USDT={self.usdt_balance:.2f}")
                    return True
                else:
                    print(f"Insufficient USDT for BUY. Needed {cost_usdt:.2f}, have {self.usdt_balance:.2f}")
                    return False

            elif trade_type == 'sell':
                if amount * price < self.min_trade_amount:
                    print(f"Trade amount too small: {amount * price:.2f} USDT (minimum: {self.min_trade_amount} USDT)")
                    return False

                if self.crypto_balance >= amount:
                    proceeds = amount * price
                    self.crypto_balance -= amount
                    self.usdt_balance += proceeds
                    trade_record = {
                        'time': trade_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'SELL',
                        'amount': amount,
                        'price': price,
                        'proceeds': proceeds
                    }
                    self.trade_history.append(trade_record)

                    if self.trading_writer:
                        self.trading_writer.add_scalar('Trades/Sell_Price', price, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Sell_Amount', amount, self.trade_step)
                        self.trading_writer.add_scalar('Trades/Sell_Proceeds', proceeds, self.trade_step)

                    print(
                        f"SELL successful. New balances: {CRYPTO_NAME}={self.crypto_balance:.{CRYPTO_DECIMALS}f}, USDT={self.usdt_balance:.2f}")
                    return True
                else:
                    print(
                        f"Insufficient {CRYPTO_NAME} for SELL. Needed {amount:.{CRYPTO_DECIMALS}f}, have {self.crypto_balance:.{CRYPTO_DECIMALS}f}")
                    return False
            else:
                print(f"Unknown trade type: {trade_type}")
                return False
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False

    def make_decision(self):
        current_time = datetime.now(UTC)
        if (current_time - self.last_trade_time).total_seconds() < DECISION_INTERVAL_SECONDS:
            return

        print(f"\nMaking decision at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.last_trade_time = current_time

        try:
            df = self._fetch_latest_data()
            if df is None or df.empty:
                print("Could not fetch data or data is empty. Skipping decision.")
                return

            df = self._calculate_indicators(df.copy())
            if df.empty or len(df) < SEQUENCE_LENGTH:
                print("Insufficient data after indicator calculation. Skipping decision.")
                return

            latest_close = df['close'].iloc[-1]
            rsi_value = df[f"RSI_{INDICATOR_CONFIG['RSI']['length']}"].iloc[-1]

            if self.trading_writer:
                self.trading_writer.add_scalar('Market/Price', latest_close, self.trade_step)
                self.trading_writer.add_scalar('Indicators/RSI', rsi_value, self.trade_step)
                self.trading_writer.add_scalar('Indicators/RSI_Overbought', INDICATOR_CONFIG['RSI']['overbought'],
                                               self.trade_step)
                self.trading_writer.add_scalar('Indicators/RSI_Oversold', INDICATOR_CONFIG['RSI']['oversold'],
                                               self.trade_step)
                self.trading_writer.add_histogram('Market/Volume', df['volume'].values, self.trade_step)

            features_tensor = self._prepare_features(df)
            if features_tensor is None:
                print("Could not prepare features. Skipping decision.")
                return

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[prediction] * 100

            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            predicted_action = action_map[prediction]

            if rsi_value >= INDICATOR_CONFIG['RSI']['overbought']:
                predicted_action = "SELL"
                print(f"RSI {rsi_value:.2f} >= {INDICATOR_CONFIG['RSI']['overbought']} (Overbought) - Forcing SELL")
            elif rsi_value <= INDICATOR_CONFIG['RSI']['oversold']:
                predicted_action = "BUY"
                print(f"RSI {rsi_value:.2f} <= {INDICATOR_CONFIG['RSI']['oversold']} (Oversold) - Forcing BUY")

            self._print_status(predicted_action, latest_close, confidence)

            if self.trading_writer:
                self.trading_writer.add_scalar('Prediction/Action', prediction, self.trade_step)
                self.trading_writer.add_scalar('Prediction/Confidence', confidence, self.trade_step)
                self.trading_writer.add_scalar('Prediction/RSI_Value', rsi_value, self.trade_step)

            trade_executed = False
            if predicted_action == "BUY":
                if latest_close > 0:
                    max_usdt_to_spend = self.usdt_balance * self.trade_percentage
                    amount_to_buy = max_usdt_to_spend / latest_close
                    trade_executed = self._execute_trade('buy', amount_to_buy, latest_close)
            elif predicted_action == "SELL":
                if latest_close > 0:
                    amount_to_sell = self.crypto_balance * self.trade_percentage
                    if amount_to_sell > 0:
                        trade_executed = self._execute_trade('sell', amount_to_sell, latest_close)

            current_value = self.crypto_balance * latest_close + self.usdt_balance
            profit_loss = current_value - self.initial_portfolio_value
            profit_loss_pct = (
                                          profit_loss / self.initial_portfolio_value) * 100 if self.initial_portfolio_value != 0 else 0

            if self.trading_writer:
                self.trading_writer.add_scalar('Portfolio/Total_Value', current_value, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Crypto_Balance', self.crypto_balance, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/USDT_Balance', self.usdt_balance, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Crypto_Value', self.crypto_balance * latest_close,
                                               self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Profit_Loss', profit_loss, self.trade_step)
                self.trading_writer.add_scalar('Portfolio/Profit_Loss_Pct', profit_loss_pct, self.trade_step)

            if trade_executed:
                print(
                    f"Trade executed. New balances: {CRYPTO_NAME}={self.crypto_balance:.{CRYPTO_DECIMALS}f}, USDT={self.usdt_balance:.2f}")
                self.trade_step += 1

        except Exception as e:
            print(f"Error in decision making: {e}")


def train_model(train_data, val_data, scaler, current_device, run_idx=0):
    num_features = train_data.features.shape[-1]
    model = CryptoTransformer(num_features=num_features, dropout_rate=DROPOUT_RATE)
    model.to(current_device)

    run_output_dir = f'./results/run_{run_idx}'
    run_logging_dir = f'./logs/run_{run_idx}'
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_dir=run_logging_dir,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        weight_decay=WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )

    train_result = trainer.train()
    eval_results = trainer.evaluate()

    model_path = os.path.join(run_output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(run_output_dir, f'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    print("\nRunning final validation...")
    val_results = trainer.evaluate()
    print(f"Final validation loss: {val_results['eval_loss']:.4f}")

    return model_path, scaler_path, val_results['eval_loss']


def train_with_multiplier_search(df, device):
    best_eval_loss = float('inf')
    best_model_info = None

    for multiplier in ATR_MULTIPLIERS_TO_TEST:
        print(f"\n=== Training with ATR Multiplier: {multiplier} ===")

        # Create targets with current multiplier
        df_temp = df.copy()
        df_temp['future_price'] = df_temp['close'].shift(-SEQUENCE_LENGTH)
        df_temp['threshold'] = df_temp[f"ATRr_{INDICATOR_CONFIG['ATR']['length']}"] * multiplier
        df_temp['target'] = 1  # HOLD
        df_temp.loc[df_temp['future_price'] > df_temp['close'] + df_temp['threshold'], 'target'] = 2  # BUY
        df_temp.loc[df_temp['future_price'] < df_temp['close'] - df_temp['threshold'], 'target'] = 0  # SELL
        df_temp.dropna(inplace=True)

        features = df_temp[FEATURES_LIST].values
        targets = df_temp['target'].values

        # Split and scale data
        split_idx = int(len(features) * 0.8)
        scaler = StandardScaler().fit(features[:split_idx])
        features = scaler.transform(features)

        # Train multiple models (per original NUM_TRAINING_RUNS)
        model_infos = []
        for run in range(NUM_TRAINING_RUNS):
            print(f"\n--- Training Run {run + 1}/{NUM_TRAINING_RUNS} ---")
            train_data = TradingDataset(features[:split_idx], targets[:split_idx], SEQUENCE_LENGTH)
            val_data = TradingDataset(features[split_idx:], targets[split_idx:], SEQUENCE_LENGTH)

            model_path, scaler_path, eval_loss = train_model(
                train_data, val_data, scaler, device,
                run_idx=f"{multiplier}_run{run}"
            )
            model_infos.append({
                'model_path': model_path,
                'scaler_path': scaler_path,
                'eval_loss': eval_loss,
                'multiplier': multiplier
            })

        # Select best model for this multiplier
        current_best = min(model_infos, key=lambda x: x['eval_loss'])
        print(f"Best for multiplier {multiplier}: Loss {current_best['eval_loss']:.4f}")

        # Track global best
        if current_best['eval_loss'] < best_eval_loss:
            best_eval_loss = current_best['eval_loss']
            best_model_info = current_best

    # Save the globally best model
    if best_model_info:
        shutil.copy(best_model_info['model_path'], BEST_MODEL_PATH)
        shutil.copy(best_model_info['scaler_path'], BEST_SCALER_PATH)
        with open(BEST_MULTIPLIER_PATH, 'w') as f:
            f.write(str(best_model_info['multiplier']))

        print(f"\nGlobal best multiplier: {best_model_info['multiplier']}")
        print(f"Best validation loss: {best_eval_loss:.4f}")

        return best_model_info['multiplier']
    return None


def save_trade_history(trader, filename='trade_history.csv'):
    try:
        if not trader.trade_history:
            print("No trade history to save")
            return

        df = pd.DataFrame(trader.trade_history)
        df.to_csv(filename, index=False)
        print(f"Trade history saved to {filename}")
    except Exception as e:
        print(f"Error saving trade history: {e}")


if __name__ == "__main__":
    current_time = datetime.now(UTC)
    print(f"Starting script at {current_time}")
    print(f"Trading pair: {PAIR}")

    TRADE_LOG_DIR = './trading_logs'
    trading_writer = SummaryWriter(log_dir=TRADE_LOG_DIR)
    print(f"TensorBoard trading logs will be saved to: {TRADE_LOG_DIR}")

    try:
        trained_model = None
        final_scaler = None
        best_multiplier = DEFAULT_ATR_MULTIPLIER

        # Try to load existing best model and scaler
        if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
            try:
                num_features_for_model = len(FEATURES_LIST)
                trained_model = CryptoTransformer(num_features=num_features_for_model, dropout_rate=DROPOUT_RATE)
                trained_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
                trained_model.to(device)
                trained_model.eval()
                final_scaler = joblib.load(BEST_SCALER_PATH)

                if os.path.exists(BEST_MULTIPLIER_PATH):
                    with open(BEST_MULTIPLIER_PATH, 'r') as f:
                        best_multiplier = float(f.read())

                print(f"Loaded existing best model (Multiplier: {best_multiplier})")
            except Exception as e:
                print(f"Error loading existing model/scaler: {e}")
                trained_model = None
                final_scaler = None

        if trained_model is None or final_scaler is None:
            print("Starting multiplier search training...")
            try:
                since_time = int((current_time - timedelta(days=LOOKBACK_DAYS_TRAINING + 7)).timestamp())
                df = api.get_ohlc_data(PAIR, interval=INTERVAL, since=since_time)[0]
                df.index = pd.to_datetime(df.index, unit='s')
                df = df[~df.index.duplicated(keep='first')]
                new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
                df = df.reindex(new_index)
                df = df.ffill().infer_objects(copy=False)

                # Calculate all indicators once
                if INDICATOR_CONFIG['RSI']['enabled']:
                    df[f"RSI_{INDICATOR_CONFIG['RSI']['length']}"] = ta.rsi(df['close'],
                                                                            length=INDICATOR_CONFIG['RSI']['length'])

                if INDICATOR_CONFIG['MACD']['enabled']:
                    macd = ta.macd(df['close'], fast=INDICATOR_CONFIG['MACD']['fast'],
                                   slow=INDICATOR_CONFIG['MACD']['slow'], signal=INDICATOR_CONFIG['MACD']['signal'])
                    df = pd.concat([df, macd], axis=1)

                if INDICATOR_CONFIG['BBANDS']['enabled']:
                    bbands = ta.bbands(df['close'], length=INDICATOR_CONFIG['BBANDS']['length'],
                                       std=INDICATOR_CONFIG['BBANDS']['std'])
                    df = pd.concat([df, bbands], axis=1)

                if INDICATOR_CONFIG['OBV']['enabled']:
                    df['OBV'] = ta.obv(df['close'], df['volume'])

                if INDICATOR_CONFIG['ADX']['enabled']:
                    adx = ta.adx(df['high'], df['low'], df['close'], length=INDICATOR_CONFIG['ADX']['length'])
                    df[f"ADX_{INDICATOR_CONFIG['ADX']['length']}"] = adx[f"ADX_{INDICATOR_CONFIG['ADX']['length']}"]

                if INDICATOR_CONFIG['CCI']['enabled']:
                    df[f"CCI_{INDICATOR_CONFIG['CCI']['length']}_{INDICATOR_CONFIG['CCI']['c']}"] = ta.cci(
                        df['high'], df['low'], df['close'],
                        length=INDICATOR_CONFIG['CCI']['length'],
                        c=INDICATOR_CONFIG['CCI']['c']
                    )

                if INDICATOR_CONFIG['ATR']['enabled']:
                    atr = ta.atr(df['high'], df['low'], df['close'], length=INDICATOR_CONFIG['ATR']['length'])
                    df[f"ATRr_{INDICATOR_CONFIG['ATR']['length']}"] = atr

                if INDICATOR_CONFIG['VWAP']['enabled']:
                    df['VWAP_D'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

                if INDICATOR_CONFIG['NATR']['enabled']:
                    df[f"NATR_{INDICATOR_CONFIG['ATR']['length']}"] = ta.natr(df['high'], df['low'], df['close'],
                                                                              length=INDICATOR_CONFIG['ATR']['length'])

                if INDICATOR_CONFIG['TRIX']['enabled']:
                    trix = ta.trix(df['close'], length=INDICATOR_CONFIG['TRIX']['length'],
                                   signal=INDICATOR_CONFIG['TRIX']['signal'])
                    df = pd.concat([df, trix], axis=1)

                if INDICATOR_CONFIG['STOCH']['enabled']:
                    stoch = ta.stoch(df['high'], df['low'], df['close'],
                                     k=INDICATOR_CONFIG['STOCH']['k'],
                                     d=INDICATOR_CONFIG['STOCH']['d'],
                                     smooth_k=INDICATOR_CONFIG['STOCH']['smooth_k'])
                    df = pd.concat([df, stoch], axis=1)

                if INDICATOR_CONFIG['EMA']['enabled']:
                    for length in INDICATOR_CONFIG['EMA']['lengths']:
                        df[f"EMA_{length}"] = ta.ema(df['close'], length=length)

                df.dropna(inplace=True)

                # Run multiplier search
                best_multiplier = train_with_multiplier_search(df, device)
                TARGET_ATR_MULTIPLIER = best_multiplier or DEFAULT_ATR_MULTIPLIER

                # Load best model
                trained_model = CryptoTransformer(num_features=len(FEATURES_LIST), dropout_rate=DROPOUT_RATE)
                trained_model.load_state_dict(torch.load(BEST_MODEL_PATH))
                trained_model.to(device)
                final_scaler = joblib.load(BEST_SCALER_PATH)

            except Exception as e:
                raise RuntimeError(f"Failed during training: {e}")

        # Initialize trader with best model
        trader = CryptoTrader(trained_model, api, final_scaler, device, live_trading=LIVE_TRADING,
                              initial_crypto_balance=INITIAL_CRYPTO_BALANCE, initial_usdt_balance=INITIAL_USDT_BALANCE,
                              trading_writer=trading_writer)

        print(f"\nStarting trading with optimal ATR multiplier: {TARGET_ATR_MULTIPLIER}")
        print("Starting trading loop... (Ctrl+C to stop)")
        try:
            while True:
                trader.make_decision()
                time.sleep(SLEEP_TIME_SECONDS)
        except KeyboardInterrupt:
            print("\nStopping trading bot...")
            save_trade_history(trader)
    finally:
        trading_writer.close()
        print("TensorBoard trading writer closed.")
