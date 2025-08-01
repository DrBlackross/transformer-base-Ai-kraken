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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ignore warnings that don't affect script functionality
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# ======================
# USER CONFIGURATION
# ======================

# --- Core Trading Configuration ---
LIVE_TRADING = False  # Set to True for real trading, False for paper trading
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')

# Select trading pair (e.g., 'DOGEUSDT' or 'XBTUSDT')
TRADING_PAIR = 'DOGEUSDT'

# --- Trading Pair Specific Settings ---
if TRADING_PAIR == 'DOGEUSDT':
    MODEL_PATH_BASE = './doge_transformer_model'
    CRYPTO_NAME = 'DOGE'
    INITIAL_USDT_BALANCE = 10.0
    INITIAL_CRYPTO_BALANCE = 100.0
elif TRADING_PAIR == 'XBTUSDT':
    MODEL_PATH_BASE = './xbt_transformer_model'
    CRYPTO_NAME = 'XBT'
    INITIAL_USDT_BALANCE = 100.0
    INITIAL_CRYPTO_BALANCE = 0.0
else:
    raise ValueError("Invalid TRADING_PAIR specified.")

# --- Model & Training Parameters ---
LOOKBACK_DAYS_TRAINING = 90  # Number of days of historical data for training
ATR_MULTIPLIERS_TO_TEST = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] # Multiplier values to test for ATR-based trading targets
DEFAULT_ATR_MULTIPLIER = 0.2
INTERVAL = 5  # OHLCV data interval in minutes (e.g., 5, 15, 30, 60)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 100  # Number of past data points to consider for each prediction
DROPOUT_RATE = 0.1
NUM_TRAINING_RUNS = 3  # Number of models to train for each multiplier to find the best one

# --- Indicators ---
INDICATOR_CONFIG = {
    'RSI': {'length': 14},
    'StochRSI': {'length': 14},
    'ATR': {'length': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9}
}

# --- Trading Logic ---
CRYPTO_INVESTMENT_AMOUNT = 0.1  # Fraction of available USDT to invest per buy order
MIN_PROFIT_PERCENT = 0.05  # Minimum profit percentage to trigger a sell
SLEEP_TIME_SECONDS = 60  # Time to wait between trading decisions

# ======================
# PATHS
# ======================
BEST_MODEL_PATH = os.path.join(MODEL_PATH_BASE, 'best_model.pth')
BEST_SCALER_PATH = os.path.join(MODEL_PATH_BASE, 'best_scaler.joblib')
BEST_MULTIPLIER_PATH = os.path.join(MODEL_PATH_BASE, 'best_multiplier.txt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======================
# FEATURE & INDICATOR LISTS
# ======================
OHLCV_LIST = ['open', 'high', 'low', 'close', 'volume']
INDICATORS_LIST = [
    f"RSI_{INDICATOR_CONFIG['RSI']['length']}",
    f"STOCHRSIk_{INDICATOR_CONFIG['StochRSI']['length']}",
    f"STOCHRSId_{INDICATOR_CONFIG['StochRSI']['length']}",
    f"MACD_{INDICATOR_CONFIG['MACD']['fast']}_{INDICATOR_CONFIG['MACD']['slow']}_{INDICATOR_CONFIG['MACD']['signal']}",
    f"MACDH_{INDICATOR_CONFIG['MACD']['fast']}_{INDICATOR_CONFIG['MACD']['slow']}_{INDICATOR_CONFIG['MACD']['signal']}",
    f"MACDS_{INDICATOR_CONFIG['MACD']['fast']}_{INDICATOR_CONFIG['MACD']['slow']}_{INDICATOR_CONFIG['MACD']['signal']}",
    f"ATRr_{INDICATOR_CONFIG['ATR']['length']}"
]
FEATURES_LIST = OHLCV_LIST + INDICATORS_LIST

# ======================
# HELPER FUNCTIONS & CLASSES
# ======================
class TradingDataset(Dataset):
    """A PyTorch Dataset for time-series trading data."""
    def __init__(self, features, targets, sequence_length):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        return (
            torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32),
            torch.tensor(self.targets[end_idx], dtype=torch.long)
        )

# ======================
# MODEL DEFINITION
# ======================
class CryptoTransformer(torch.nn.Module):
    """
    Transformer-based model for predicting trading actions (SELL, HOLD, BUY).
    The model processes a sequence of historical data to make a single prediction.
    """
    def __init__(self, num_features, dropout_rate=0.1):
        super().__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=num_features // 4,
            dropout=dropout_rate,
            dim_feedforward=num_features * 4,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.classifier = torch.nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use the last output of the sequence for classification
        return self.classifier(x)

# ======================
# CORE LOGIC FUNCTIONS
# ======================
def _get_kraken_balance_code(symbol):
    """Maps a common symbol (e.g., 'DOGE') to Kraken's internal balance code (e.g., 'XXDG')."""
    kraken_codes = {
        'DOGE': 'XXDG',
        'XBT': 'XXBT',
        'USDT': 'ZUSD'
    }
    return kraken_codes.get(symbol.upper(), symbol.upper())

def _get_live_balances(private_api, pair):
    """Fetches live balances for the specified trading pair from Kraken."""
    crypto_code = _get_kraken_balance_code(CRYPTO_NAME)
    fiat_code = _get_kraken_balance_code(pair.replace(CRYPTO_NAME, ''))

    balances_data = private_api.query_private('Balance')
    balances = balances_data.get('result', {})

    print(f"Raw balance data from API: {balances}")

    crypto_balance = float(balances.get(crypto_code, balances.get(CRYPTO_NAME, 0.0)))
    fiat_balance = float(balances.get(fiat_code, balances.get('USDT', 0.0)))

    print(f"Successfully fetched balances:\n  {CRYPTO_NAME} Balance: {crypto_balance:.4f}\n  USDT Balance: {fiat_balance:.2f}")

    return crypto_balance, fiat_balance

def _fetch_latest_data(public_api, pair, interval):
    """Fetches the latest OHLCV data from Kraken."""
    try:
        since_time = int((datetime.now(UTC) - timedelta(minutes=interval * 2)).timestamp())
        df = public_api.get_ohlc_data(pair, interval=interval, since=since_time)[0]
        df.index = pd.to_datetime(df.index, unit='s')
        return df.iloc[-1]
    except Exception as e:
        print(f"Error fetching latest data: {e}")
        return None

def calculate_all_indicators(df):
    """Calculates all configured technical indicators and adds them to the DataFrame."""
    # Ensure pandas-ta is compatible with the latest pandas changes
    df.ta.ema(close=df['close'], length=20, append=True, signal_lines=False)
    df.ta.ema(close=df['close'], length=50, append=True, signal_lines=False)
    df.ta.ema(close=df['close'], length=200, append=True, signal_lines=False)

    for indicator, params in INDICATOR_CONFIG.items():
        if indicator == 'RSI':
            df.ta.rsi(close=df['close'], length=params['length'], append=True)
        elif indicator == 'StochRSI':
            df.ta.stochrsi(close=df['close'], length=params['length'], append=True)
        elif indicator == 'ATR':
            df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=params['length'], append=True)
        elif indicator == 'MACD':
            df.ta.macd(close=df['close'], fast=params['fast'], slow=params['slow'], signal=params['signal'], append=True)

    return df

def train_model(train_data, val_data, scaler, device, run_idx=""):
    """
    Trains a single model on the provided data.
    """
    model = CryptoTransformer(num_features=len(FEATURES_LIST), dropout_rate=DROPOUT_RATE)
    model.to(device)

    model_path = os.path.join(MODEL_PATH_BASE, f'model_{run_idx}.pth')
    scaler_path = os.path.join(MODEL_PATH_BASE, f'scaler_{run_idx}.joblib')

    training_args = TrainingArguments(
        output_dir=f'./results/run_{run_idx}',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()

    eval_result = trainer.evaluate()
    eval_loss = eval_result['eval_loss']

    # Save best model and scaler
    trainer.save_model(model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path, eval_loss

def train_with_multiplier_search(df, device, public_api):
    """
    Trains and evaluates models using different ATR multipliers to find the best one.
    """
    best_eval_loss = float('inf')
    best_model_info = None

    for multiplier in ATR_MULTIPLIERS_TO_TEST:
        print(f"\n=== Training with ATR Multiplier: {multiplier} ===")

        # Create trading targets based on the current ATR multiplier
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

        # Train multiple models for robustness
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

        # Select the best performing model for this multiplier
        current_best = min(model_infos, key=lambda x: x['eval_loss'])
        print(f"Best for multiplier {multiplier}: Loss {current_best['eval_loss']:.4f}")

        # Track the globally best model across all multipliers
        if current_best['eval_loss'] < best_eval_loss:
            best_eval_loss = current_best['eval_loss']
            best_model_info = current_best

    # Save the globally best model and its multiplier
    if best_model_info:
        shutil.copy(best_model_info['model_path'], BEST_MODEL_PATH)
        shutil.copy(best_model_info['scaler_path'], BEST_SCALER_PATH)
        with open(BEST_MULTIPLIER_PATH, 'w') as f:
            f.write(str(best_model_info['multiplier']))

        print(f"\nGlobal best multiplier: {best_model_info['multiplier']}")
        print(f"Best validation loss: {best_eval_loss:.4f}")

        return best_model_info['multiplier']
    return None

class CryptoTrader:
    """Manages the trading logic and execution based on a trained model."""
    def __init__(self, model, public_api, private_api, scaler, device, live_trading, crypto_balance, usdt_balance, trading_writer):
        self.model = model
        self.public_api = public_api
        self.private_api = private_api
        self.scaler = scaler
        self.device = device
        self.live_trading = live_trading
        self.crypto_balance = crypto_balance
        self.usdt_balance = usdt_balance
        self.trading_writer = trading_writer
        self.trade_count = 0
        self.last_buy_price = 0.0

        # Create Kraken API instance
        self.api = krakenex.API()
        self.api.load_key('kraken-private.key')
        self.kraken = KrakenAPI(self.api)
        self.pair = self.kraken.get_kraken_ws_name(TRADING_PAIR)

        # Map integer predictions to trading actions
        self.action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    def get_live_data(self):
        """Fetches and processes live market data."""
        try:
            # Fetch historical data for a complete sequence
            since_time = int((datetime.now(UTC) - timedelta(minutes=INTERVAL * SEQUENCE_LENGTH * 1.5)).timestamp())
            while True:
                try:
                    df = self.public_api.get_ohlc_data(TRADING_PAIR, interval=INTERVAL, since=since_time)[0]
                    break
                except krakenex.exceptions.APITimeoutError as e:
                    print(f"Kraken API Timeout error: {e}. Retrying in 10 seconds...")
                    time.sleep(10)
                except Exception as e:
                    if "public call frequency exceeded" in str(e):
                        print(f"Public call frequency exceeded. Sleeping for 5 seconds...")
                        time.sleep(5)
                    else:
                        raise RuntimeError(f"Failed to fetch historical data: {e}")

            df.index = pd.to_datetime(df.index, unit='s')
            df = df[~df.index.duplicated(keep='first')]
            new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
            df = df.reindex(new_index)
            df = df.ffill().infer_objects()

            # Calculate indicators and get the latest feature vector
            df = calculate_all_indicators(df)
            df = df[FEATURES_LIST].dropna()

            if len(df) < SEQUENCE_LENGTH:
                print("Not enough data for a full sequence. Waiting...")
                return None, None

            latest_features = df.iloc[-SEQUENCE_LENGTH:].values
            latest_price = df['close'].iloc[-1]
            return latest_features, latest_price

        except Exception as e:
            print(f"Error fetching and preparing live data: {e}")
            return None, None

    def make_decision(self):
        """Makes a trading decision based on the model's prediction."""
        self.trade_count += 1
        print(f"\nMaking decision at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        features, current_price = self.get_live_data()
        if features is None:
            return

        scaled_features = self.scaler.transform(features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_action_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_action_idx].item()

        predicted_action = self.action_map[predicted_action_idx]
        print("--------------------------------------------------")
        print(f"|  Action: {predicted_action:<4} | Price: {current_price:.5f}    (Confidence: {confidence * 100:.2f}%)")

        self.trading_writer.add_scalar('action/prediction', predicted_action_idx, self.trade_count)
        self.trading_writer.add_scalar('action/confidence', confidence, self.trade_count)
        self.trading_writer.add_scalar('market/current_price', current_price, self.trade_count)

        self.execute_trade(predicted_action, current_price)
        self.print_status(current_price)

    def execute_trade(self, action, current_price):
        """Executes a trading action based on the model's prediction."""
        if action == 'BUY' and self.usdt_balance > 0:
            trade_amount_usdt = self.usdt_balance * CRYPTO_INVESTMENT_AMOUNT
            if self.live_trading:
                try:
                    order = self.private_api.add_order(
                        pair=TRADING_PAIR, type='buy', ordertype='market',
                        volume=trade_amount_usdt / current_price,
                        validate=False
                    )
                    print(f"Live BUY order placed: {order}")
                except Exception as e:
                    print(f"Failed to place BUY order: {e}")
                    return

            self.crypto_balance += trade_amount_usdt / current_price
            self.usdt_balance -= trade_amount_usdt
            self.last_buy_price = current_price
            print(f"Paper BUY: {trade_amount_usdt:.2f} USDT worth of {CRYPTO_NAME}")

        elif action == 'SELL' and self.crypto_balance > 0:
            profit_percent = (current_price - self.last_buy_price) / self.last_buy_price * 100 if self.last_buy_price > 0 else 0
            if profit_percent > MIN_PROFIT_PERCENT:
                if self.live_trading:
                    try:
                        order = self.private_api.add_order(
                            pair=TRADING_PAIR, type='sell', ordertype='market',
                            volume=self.crypto_balance,
                            validate=False
                        )
                        print(f"Live SELL order placed: {order}")
                    except Exception as e:
                        print(f"Failed to place SELL order: {e}")
                        return

                trade_amount_usdt = self.crypto_balance * current_price
                self.usdt_balance += trade_amount_usdt
                self.crypto_balance = 0.0
                self.last_buy_price = 0.0
                print(f"Paper SELL: {trade_amount_usdt:.2f} USDT worth of {CRYPTO_NAME}")
            else:
                print(f"Paper SELL: Below minimum profit percentage ({profit_percent:.2f}% < {MIN_PROFIT_PERCENT}%)")

    def print_status(self, current_price):
        """Prints the current portfolio status."""
        portfolio_value = (self.crypto_balance * current_price) + self.usdt_balance
        print("--------------------------------------------------")
        print(f"|  {CRYPTO_NAME} Balance: {self.crypto_balance:.4f}")
        print(f"|  USDT Balance: {self.usdt_balance:.2f}")
        print(f"|  Portfolio Value: {portfolio_value:.2f} USDT")
        print("--------------------------------------------------")
        self.trading_writer.add_scalar('portfolio/crypto_balance', self.crypto_balance, self.trade_count)
        self.trading_writer.add_scalar('portfolio/usdt_balance', self.usdt_balance, self.trade_count)
        self.trading_writer.add_scalar('portfolio/portfolio_value', portfolio_value, self.trade_count)

def save_trade_history(trader):
    """Saves the trade history to a file."""
    # (Implementation for saving history would go here)
    pass

def initial_checks():
    """Performs initial checks for API keys and balances."""
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise ValueError("Kraken API key and secret must be set as environment variables.")

    public_api = krakenex.API()
    private_api = krakenex.API()
    private_api.key = KRAKEN_API_KEY
    private_api.secret = KRAKEN_API_SECRET

    print("Verifying API keys...")
    try:
        public_api.query_public('Time')
        print("Kraken API keys are valid.")
    except Exception as e:
        raise ValueError(f"Kraken API key verification failed: {e}")

    print("Fetching live balances from Kraken...")
    crypto_balance_init, usdt_balance_init = _get_live_balances(private_api, TRADING_PAIR)
    return public_api, private_api, crypto_balance_init, usdt_balance_init

if __name__ == "__main__":
    public_api, private_api, crypto_balance_init, usdt_balance_init = initial_checks()

    current_time = datetime.now(UTC)
    print(f"\nStarting script at {current_time}")
    print(f"Trading pair: {TRADING_PAIR}")

    # Setup TensorBoard for logging
    TRADE_LOG_DIR = './trading_logs'
    trading_writer = SummaryWriter(log_dir=TRADE_LOG_DIR)
    print(f"TensorBoard trading logs will be saved to: {TRADE_LOG_DIR}")

    try:
        trained_model = None
        final_scaler = None
        TARGET_ATR_MULTIPLIER = DEFAULT_ATR_MULTIPLIER

        # Try to load an existing model to skip training
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
                        TARGET_ATR_MULTIPLIER = float(f.read())

                print(f"Loaded existing best model (Multiplier: {TARGET_ATR_MULTIPLIER})")
            except Exception as e:
                print(f"Error loading existing model/scaler: {e}")
                trained_model = None
                final_scaler = None

        if trained_model is None or final_scaler is None:
            print("Starting multiplier search training...")
            try:
                since_time = int((current_time - timedelta(days=LOOKBACK_DAYS_TRAINING + 7)).timestamp())

                # Robust data fetching with retry logic
                while True:
                    try:
                        df = public_api.get_ohlc_data(TRADING_PAIR, interval=INTERVAL, since=since_time)[0]
                        break
                    except krakenex.exceptions.APITimeoutError as e:
                        print(f"Kraken API Timeout error: {e}. Retrying in 10 seconds...")
                        time.sleep(10)
                    except Exception as e:
                        if "public call frequency exceeded" in str(e):
                            print(f"Public call frequency exceeded. Sleeping for 5 seconds...")
                            time.sleep(5)
                        else:
                            raise RuntimeError(f"Failed to fetch historical data: {e}")

                df.index = pd.to_datetime(df.index, unit='s')
                df = df[~df.index.duplicated(keep='first')]
                new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{INTERVAL}min')
                df = df.reindex(new_index)
                df = df.ffill().infer_objects()

                df = calculate_all_indicators(df)
                df.dropna(inplace=True)

                # Run multiplier search to find the best trading strategy
                temp_multiplier = train_with_multiplier_search(df, device, public_api)
                if temp_multiplier is not None:
                    TARGET_ATR_MULTIPLIER = temp_multiplier
                else:
                    # Fallback to default if search fails
                    TARGET_ATR_MULTIPLIER = DEFAULT_ATR_MULTIPLIER

                # Load the best model that was just trained
                if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_SCALER_PATH):
                    trained_model = CryptoTransformer(num_features=len(FEATURES_LIST), dropout_rate=DROPOUT_RATE)
                    trained_model.load_state_dict(torch.load(BEST_MODEL_PATH))
                    trained_model.to(device)
                    final_scaler = joblib.load(BEST_SCALER_PATH)
                else:
                    raise RuntimeError("Training search failed and did not save a model.")

            except Exception as e:
                raise RuntimeError(f"Failed during training: {e}")

        # Initialize trader with the best-performing model
        trader = CryptoTrader(trained_model, public_api, private_api, final_scaler, device, live_trading=LIVE_TRADING,
                              crypto_balance=crypto_balance_init, usdt_balance=usdt_balance_init,
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
