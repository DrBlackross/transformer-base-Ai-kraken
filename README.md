(i've been lurking on github for years, this is my best attempt at replacing all the old crypto strategy based scripts (for my needs) since askmike came out with Gekko... here is my take, feel free to muck with it just give credit, any issues, you can post but i will probably never see it LOL (i do know this works well with kraken, coinbase being what the api is, its touchy))

# Kraken Transformer Trading Bot

<img width="1760" height="688" alt="image" src="https://github.com/user-attachments/assets/d4340160-823b-4df2-9cfa-10bbceb6a0aa" />


This project implements an advanced cryptocurrency trading bot designed for the Kraken exchange. It leverages a Transformer neural network for predictive analysis, integrates various technical indicators, and supports both paper and live trading modes. A core feature of this bot is its Adaptive ATR (Average True Range) Multiplier Optimization, which dynamically fine-tunes trade thresholds for improved strategy performance. Project Overview

## **The bot operates by:**

  - **Historical Data Acquisition:** Pulling approximately two years of historical candlestick data from Kraken for the selected trading pair.

-   **Model Training & Selection:** Training multiple Transformer models based on this extensive dataset. It then automatically selects the best-performing model (based on validation loss) and its corresponding optimal ATR multiplier.

- **Market Monitoring & Decision Making:** Continuously monitoring the market (checking every 5 minutes by default) and using the trained model to predict future price movements (Buy, Hold, Sell actions).

-   **Portfolio Profit Seeking:** The primary objective of the bot's strategy is to seek overall portfolio profit. It aims to make decisions that lead to capital appreciation over time.

While it might not generate "major profits" instantly, the goal is consistent, positive performance tailored for the Kraken market. Features

-   **Transformer Model:** Utilizes a custom Transformer neural network to predict future price actions (Buy, Hold, Sell).

 -  **Comprehensive Technical Indicators:** Integrates a wide range of pandas-ta indicators (RSI, MACD, Bollinger Bands, OBV, ADX, CCI, ATR, VWAP, NATR, TRIX, STOCH, EMA) for robust market analysis.

-   **Adaptive ATR Multiplier Optimization:** Automatically searches and selects the most effective Average True Range (ATR) multiplier during training. This multiplier defines the volatility-based thresholds for trade signals, adapting the strategy to market conditions.

-  **Paper Trading Mode:** Allows for risk-free simulation of trading strategies using real market data.

-   **Live Trading Mode:** Capable of executing actual trades on the Kraken exchange (requires secure API key configuration).

-   **Highly Configurable:** Easily adjust trading pairs (DOGEUSDT or XBTUSDT), initial balances, minimum trade amounts, model training parameters (e.g., lookback periods, sequence length, epochs), and individual technical indicator settings.

-   **TensorBoard Integration:** Provides detailed logging of both model training progress and live trading performance metrics for visual analysis.

-   **Detailed Trade History:** Automatically saves a CSV file documenting all executed trades for post-analysis.



###   **Getting Started**

Follow these steps to set up and run the Kraken Transformer Trading Bot. Prerequisites

Python 3.8+ installed on your system.

A Kraken API account (for live trading, generate API keys with Fund and Trade permissions, same with CoinBase).

An active internet connection to fetch historical and live market data from Kraken.

**1.** **Clone the Repository**

First, clone this GitHub repository to your local machine:

	git clone [https://github.com/your-username/kraken-trans-bot.git](https://github.com/DrBlackross/transformer-base-Ai-kraken.git) ; cd transformer-base-Ai-kraken

**2.** **Set Up a Python Virtual Environment**
It is highly recommended to use a Python virtual environment to manage project dependencies and avoid conflicts with your system's Python installation. For Linux/macOS:

	python3 \-m venv venv source venv/bin/activate

For Windows (Command Prompt):

	python \-m venv venv venv\\Scripts\\activate.bat

For Windows (PowerShell):

	python \-m venv venv .\\venv\\Scripts\\Activate.ps1

**3.** **Install Dependencies**

Once your virtual environment is activated, install the required packages using the requirements\_x86.txt file:

	pip install \-r requirements\_x86.txt

This command will install all necessary libraries, including torch, transformers, pandas, krakenex, pandas-ta, and tensorboard. 

**4.** **Configure API Keys (for Live Trading)**

If you intend to use live trading (LIVE\_TRADING \= True), you must set your Kraken API Key and Secret as environment variables. The script kraken-trans-bot.py securely retrieves these using os.getenv().

Important: Never hardcode your API keys directly into the script files (unless directly testing). 

For Linux/macOS (add to your \~/.bashrc, \~/.zshrc, or equivalent):

export KRAKEN\_API\_KEY="YOUR\_KRAKEN\_API\_KEY" export KRAKEN\_API\_SECRET="YOUR\_KRAKEN\_API\_SECRET"

------------


(either way if using coinbase api, ugh)

It should look like this if you hard code the keys in the script right

COINBASE\_API\_KEY \= os.getenv(organizations/COINBASE\_KEY\_SHOULD\_BE\_LIKE\_THIS/apiKeys/AND\_THE\_REST\_OF\_THE\_KEY)

COINBASE\_API\_SECRET \= os.getenv(\-----BEGIN EC PRIVATE KEY-----\\FUN\_PART\_OF\_COINBASE\_NIGHTMARE\_API\_SECRET\\n\-----END EC PRIVATE KEY-----\\n)  

(if that doesn't work try adding ‘ ‘ at the begining and end) 

COINBASE\_API\_KEY \= os.getenv(‘organizations/COINBASE\_KEY\_SHOULD\_BE\_LIKE\_THIS/apiKeys/AND\_THE\_REST\_OF\_THE\_KEY’)

COINBASE\_API\_SECRET \= os.getenv(‘\-----BEGIN EC PRIVATE KEY-----\\FUN\_PART\_OF\_COINBASE\_NIGHTMARE\_API\_SECRET\\n\-----END EC PRIVATE KEY-----\\n’)

But, I know for sure hardcoding worked for testing, coinbase api is bit *weird* (if someone knows how to .env this please fork it and just give credit lol)

------------


After adding these, either run source \~/.bashrc (or your respective file) or open a new terminal session for the changes to take effect. For Windows (Command Prompt \- temporary for the current session):

set KRAKEN\_API\_KEY="YOUR\_KRAKEN\_API\_KEY" set KRAKEN\_API\_SECRET="YOUR\_KRAKEN\_API\_SECRET"

For Windows (PowerShell \- temporary for the current session):

$env:KRAKEN\_API\_KEY="YOUR\_KRAKEN\_API\_KEY" $env:KRAKEN\_API\_SECRET="YOUR\_KRAKEN\_API\_SECRET"

For persistent environment variables on Windows, you will need to add them via the System Properties \-\> Environment Variables dialog. 


**5.** **Customize Bot Settings**

 Open the kraken-trans-bot.py file in your preferred code editor and navigate to the "USER CONFIGURABLE SETTINGS" section near the top.

#### **Key parameters you might want to adjust include (for fun IN PAPER):**

 LIVE\_TRADING: Set to False for paper trading (highly recommended for initial testing), or True for live trading (see yada-yada at bottom of README).

 TRADING\_PAIR: Select your desired cryptocurrency pair, e.g., 'DOGEUSDT' or 'XBTUSDT' (Kraken) OR 'DOGE/USDT' or 'BTC/USDT' (CoinBase).

 INITIAL\_USDT\_BALANCE / INITIAL\_CRYPTO\_BALANCE: These define your starting capital for paper trading simulations. For live trading, your actual exchange balances will be used.

 LOOKBACK\_DAYS\_TRAINING: Determines the amount of historical data (in days) used to train the prediction model (only went as far back as 4 years, default is 2 years).

 SEQUENCE\_LENGTH: Represents the number of past time steps (e.g., minutes of candle data) the Transformer model analyzes for each prediction.

 NUM\_TRAINING\_RUNS: The number of individual training sessions performed for each ATR multiplier value during optimization. Reducing this can speed up initial setup (you can go higher than 5, but the model will plateau or overfit data).

 NUM\_TRAIN\_EPOCHS: The maximum number of training epochs per model. Consider lowering this for quicker testing cycles (best is what its at just leave it).

 ATR\_MULTIPLIERS\_TO\_TEST: A list of volatility multipliers that the bot will evaluate to find the most effective trading threshold (have fun with this, it runs all and find the most ‘well trained’ model). 

**6.** **Running the Bot**

 With your virtual environment activated and settings configured, execute the bot from your terminal:

	python kraken-trans-bot.py

 The bot will first either load a previously saved optimal model and scaler (if available) or initiate a new training process, which includes the ATR multiplier search. Once training is complete (or a model is loaded), it will enter its continuous trading loop, making decisions at the DECISION\_INTERVAL\_SECONDS interval defined in your settings.

- Manual Monitoring: You would need to manually observe the market and start the script by running python kraken-trans-bot.py when you believe the market is at an opportune low.

 Simply stop the bot at any time by pressing Ctrl+C. This action will trigger a shutdown routine that includes saving your trade history. You should start the bot at the lowest price point of the day (what you think it is). If you wish to initiate the bots operation at a specific market condition, start like this…

	rm ./trading\_logs/\*; rm ./crypto\_transformer\_model\_best.pth; rm ./standard\_scaler\_best.pkl; rm \-rf ./logs/\*; rm \-rf ./results/\*; python ./[kraken-trans-bot.py](http://kraken-trans-bot.py)

 This will clear all the previous logs and retrain the model, sometimes this helpful.

 Initial Holdings: The INITIAL\_CRYPTO\_BALANCE and INITIAL\_USDT\_BALANCE settings are crucial for paper trading. If you want to simulate starting with a specific asset distribution at a "low point," adjust these values accordingly before launching in paper mode.

**7. ** **Monitoring with TensorBoard**

 The bot extensively logs its training metrics and live trading performance, which can be visualized using TensorBoard. To access these logs, open a new terminal window (I run screen for this in linux), activate your Python virtual environment, and run:

	tensorboard \--logdir=./logs \--bind\_all
 
(sometimes it does not work, try this)

	pkill -f tensorboard
	tensorboard --logdir=./logs --logdir=./trading_logs --port=6006 --host 0.0.0.0

 Then, open your web browser and navigate to the address provided by TensorBoard (typically [http://localhost:6006](http://localhost:6006)). You will find distinct log directories for:

./logs/run\_\*: Detailed metrics from each model training run.

./trading\_logs: Comprehensive performance metrics and trade events from the live (or paper) trading sessions.

**8.**  **Analyzing Trade History**

 Upon a shutdown of the bot (via Ctrl+C), a trade\_history.csv file will be generated or updated in the projects root directory. This file contains a detailed record of all transactions executed during that session, including trade type, amount, price, and timestamp.

(the usual yada-yada) 

**Disclaimer**: Cryptocurrency trading carries substantial financial risk, including the potential for total loss of capital. This trading bot is provided for educational, research, and experimental purposes only. It is not financial advice. Do not use this bot for live trading with funds you cannot afford to lose. Always thoroughly understand the code, test extensively in paper trading mode, and fully comprehend the risks involved before deploying any automated trading system with real money. The author assumes no responsibility for any financial losses incurred through the use of this software.

**If it works or you think its cool, send me a few Dogecoin  DNgPXztNRmj5qp5jdPP2ZKm1r4u6eQmaZJ**
