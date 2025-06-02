import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta  # pip install ta
import yfinance as yf

# ====== CONFIGURATION ======
SEED = 5           # Change this for each model (1, 2, 3, ...)
MODEL_NAME = f"model{SEED}.h5"
SYMBOL = "EURUSD=X"    # Edit to match your trading symbol
EPOCHS = 25
BATCH_SIZE = 32
WINDOW_SIZE = 60       # Number of minutes in each sequence

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ====== DATA FETCH & FEATURE ENGINEERING ======
print("Downloading data...")
# Yahoo only allows 7 days of 1m data!
df = yf.download(SYMBOL, interval="1m", period="7d")
df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
df = df.dropna()

# Always use 1D Series for indicators (avoid shape errors)
close = df['close'].squeeze()
high = df['high'].squeeze()
low = df['low'].squeeze()

# Technical indicators
df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
macd = ta.trend.MACD(close)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()
stoch = ta.momentum.StochasticOscillator(high, low, close)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()
df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
df['sma_7'] = ta.trend.SMAIndicator(close, window=7).sma_indicator()
df['sma_25'] = ta.trend.SMAIndicator(close, window=25).sma_indicator()
df['sma_99'] = ta.trend.SMAIndicator(close, window=99).sma_indicator()

df = df.dropna()
feature_cols = ['close','rsi','macd','macd_signal','macd_diff','stoch_k','stoch_d','atr','sma_7','sma_25','sma_99']
features = df[feature_cols].values

# ====== LABEL CREATION (NEXT CANDLE UP/DOWN) ======
labels = (df['close'].shift(-1) > df['close']).astype(int)[:-1].values  # Ensure labels is a NumPy array
features = features[:-1]

# ====== CREATE SEQUENCES ======
X, y = [], []
for i in range(WINDOW_SIZE, len(features)):
    X.append(features[i-WINDOW_SIZE:i])
    y.append(labels[i])
X, y = np.array(X), np.array(y)

# ====== TRAIN/VAL SPLIT ======
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# ====== MODEL DEFINITION ======
model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ====== TRAIN ======
print("Training model...")
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])

# ====== SAVE ======
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# ====== OPTIONAL: QUICK TEST ======
score = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {score[1]*100:.2f}%")
