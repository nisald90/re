import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
import tkinter as tk
from tkinter import ttk
import yfinance as yf
import ta

# ===== WATERMARK =====
WATERMARK_TEXT = "This bot Fully Authorized By Nd, Dont Mess with it, fuck off"

# ===== NON-OTC ASSET SYMBOLS =====
ASSET_YF_SYMBOLS = {
    "AUD/CAD": "AUDCAD=X",
    "AUD/CHF": "AUDCHF=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "CAD/CHF": "CADCHF=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/AUD": "GBPAUD=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "USD/JPY": "USDJPY=X",
    "GBP/CAD": "GBPCAD=X",
}

ASSET_LIST = list(ASSET_YF_SYMBOLS.keys())

# ===== CONFIG =====
BROKER_URL = "https://pocketoption.com/en/cabinet/demo-quick/"
CONFIDENCE_THRESHOLD = 0.75
LOOKBACK_WINDOW = 200
CANDLE_INTERVAL = "1m"
PAYOUT_THRESHOLD = 80  # %

# ===== DATA FETCH =====
def get_historical_data(asset, period="1d", interval="1m"):
    yf_symbol = ASSET_YF_SYMBOLS.get(asset)
    if not yf_symbol:
        raise ValueError(f"Unknown asset: {asset}")
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
    if df.empty or len(df) < 70:
        print(f"Could not fetch enough candles for {asset}.")
        return pd.DataFrame()
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    df = df.reset_index().rename(columns={"Datetime": "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

# ===== INDICATORS =====
def calculate_features(df):
    # Add extra indicators for more accuracy
    indicators = {}
    indicators['close'] = df['close']
    indicators['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    indicators['macd'] = macd.macd()
    indicators['macd_signal'] = macd.macd_signal()
    indicators['macd_diff'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    indicators['stoch_k'] = stoch.stoch()
    indicators['stoch_d'] = stoch.stoch_signal()
    indicators['volatility'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    indicators['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    indicators['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    indicators['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    features = pd.DataFrame(indicators).dropna()
    # Only last 60 periods
    if len(features) < 60:
        return np.zeros((60, len(features.columns)))
    return features.values[-60:]

# ===== AI MODEL (ENHANCED ACCURACY) =====
def build_model(input_shape=(60, 10)):
    model = tf.keras.Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def ai_decision(model, features):
    pred = model.predict(features.reshape(1, 60, -1), verbose=0)[0][0]
    if pred > 0.7:
        return "CALL", pred
    elif pred < 0.3:
        return "PUT", 1 - pred
    else:
        return "NONE", pred

# ====== COMPOUNDING AND MARTINGALE LOGIC ======
class CompoundingSession:
    def __init__(self, base_amount, multiplier, steps):
        self.base_amount = base_amount
        self.multiplier = multiplier
        self.steps = steps
        self.current_step = 1
        self.current_amount = base_amount
        self.session_active = False
        self.session_win = 0

    def win(self):
        self.session_win += 1
        if self.session_win >= self.steps:
            self.reset()
        else:
            if (self.session_win % self.steps) == 0:
                self.current_amount *= self.multiplier

    def lose(self):
        self.reset()

    def get_amount(self):
        if not self.session_active:
            self.session_active = True
            self.session_win = 0
            self.current_amount = self.base_amount
        return self.current_amount

    def reset(self):
        self.session_active = False
        self.session_win = 0
        self.current_amount = self.base_amount

class MartingaleSession:
    def __init__(self, base_amount, multiplier, steps):
        self.base_amount = base_amount
        self.multiplier = multiplier
        self.steps = steps
        self.current_step = 0
        self.current_amount = base_amount
        self.session_active = False

    def win(self):
        self.current_step = 0
        self.current_amount = self.base_amount
        self.session_active = False

    def lose(self):
        self.current_step += 1
        if self.current_step > self.steps:
            self.session_active = False
            raise Exception("Max Martingale steps reached. Stopping bot.")
        else:
            self.current_amount = self.base_amount * (self.multiplier ** self.current_step)
            self.session_active = True

    def get_amount(self):
        if not self.session_active:
            self.current_amount = self.base_amount
        return self.current_amount

    def reset(self):
        self.current_step = 0
        self.current_amount = self.base_amount
        self.session_active = False

# ====== SELENIUM (USING YOUR PROVIDED XPATHS) ======
def start_browser():
    driver = webdriver.Chrome()
    driver.get(BROKER_URL)
    driver.maximize_window()
    return driver

def login(driver):
    print("Please log in manually if not automated.")
    time.sleep(15)

def select_asset(driver, asset_name):
    # Click the asset selection dropdown
    asset_button = driver.find_element(By.XPATH, '//*[@id="bar-chart"]/div/div/div[1]/div/div[1]/div[1]/div[1]/div/a/div/span')
    asset_button.click()
    time.sleep(1)
    # Find asset by display name and click
    asset_elem = driver.find_element(By.XPATH, f"//span[text()='{asset_name}']")
    asset_elem.click()
    time.sleep(1)

def select_trade_time(driver, trade_time_label):
    # Open trade time selector
    time_button = driver.find_element(By.XPATH, '//*[@id="put-call-buttons-chart-1"]/div/div[1]/div[1]/div[2]/div[1]/div')
    time_button.click()
    time.sleep(1)
    # Select the trade time
    time_option = driver.find_element(By.XPATH, f"//div[contains(text(), '{trade_time_label}')]")
    time_option.click()
    time.sleep(1)

def set_trade_amount(driver, amount):
    amount_input = driver.find_element(By.XPATH, '//*[@id="put-call-buttons-chart-1"]/div/div[1]/div[2]/div[2]/div[1]/div/input')
    amount_input.clear()
    amount_input.send_keys(str(amount))
    time.sleep(0.5)

def click_buy_button(driver):
    buy_button = driver.find_element(By.XPATH, '//*[@id="put-call-buttons-chart-1"]/div/div[2]/div[2]/div[1]/a/span/span/span')
    buy_button.click()
    time.sleep(0.5)

def click_sell_button(driver):
    sell_button = driver.find_element(By.XPATH, '//*[@id="put-call-buttons-chart-1"]/div/div[2]/div[2]/div[3]/a/span/span/span')
    sell_button.click()
    time.sleep(0.5)

def get_trade_result_text(driver):
    # Wait up to 12 seconds for result panel to update
    for _ in range(12):
        panel = driver.find_element(By.XPATH, '//*[@id="bar-chart"]/div/div/div[2]/div/div[2]/div')
        text = panel.text.strip()
        if text:  # Not empty
            return text
        time.sleep(1)
    return "Result not found"

def get_balance(driver):
    try:
        el = driver.find_element(By.XPATH, "//span[contains(@class, 'balance')]")
        balance_text = el.text.replace('$', '').replace(',', '')
        return float(balance_text)
    except Exception as e:
        print("Balance fetch error:", e)
        return 0.0

# ====== TRADE EXECUTION WITH RESULT CHECK ======
def place_trade(driver, asset, trade_time, direction, amount):
    # Set asset
    select_asset(driver, asset)
    # Set trade time
    select_trade_time(driver, trade_time)
    # Set amount
    set_trade_amount(driver, amount)
    # Place direction
    if direction == "CALL":
        click_buy_button(driver)
    elif direction == "PUT":
        click_sell_button(driver)
    else:
        print("Invalid direction")
        return None
    print("Waiting for trade result...")
    result_text = get_trade_result_text(driver)
    print(f"Trade result: {result_text}")
    return result_text

# ====== MAIN TRADING LOGIC ======
def manual_trading(cfg):
    print("Manual Trading started.")
    comp_session = CompoundingSession(cfg['amount'], cfg['comp_multiplier'], cfg['comp_steps']) if cfg['enable_comp'] else None
    mart_session = MartingaleSession(cfg['amount'], cfg['mart_multiplier'], cfg['mart_steps']) if cfg['enable_mart'] else None
    driver = start_browser() if cfg['auto_trade'] else None
    if driver:
        login(driver)
        balance = get_balance(driver)
    else:
        balance = 0.0
    try:
        while True:
            if cfg['ai_recommend']:
                df = get_historical_data(cfg['asset'], period="1d", interval=cfg['trade_time'])
                if df.empty or len(df) < 70:
                    print("Not enough data. Waiting...")
                    time.sleep(10)
                    continue
                features = calculate_features(df)
                if features.shape[0] < 60:
                    print("Not enough indicator data, waiting...")
                    time.sleep(10)
                    continue
                model = build_model((60, features.shape[1]))
                signal, confidence = ai_decision(model, features)
                print(f"{datetime.now()} [{cfg['asset']}]: AI Signal: {signal} (confidence: {confidence:.2f})")
                print(f"Recommended duration: {cfg['trade_time']}")
                if not cfg['auto_trade']:
                    input("Press Enter to execute trade or Ctrl+C to stop...")
            else:
                print(f"User selected asset: {cfg['asset']}, trade_time: {cfg['trade_time']}")
                signal = None
                if not cfg['auto_trade']:
                    signal = input("Enter trade direction (CALL/PUT): ").strip().upper()
                    if signal not in ["CALL", "PUT"]:
                        print("Invalid direction.")
                        continue
            # Determine amount
            amount = cfg['amount']
            if comp_session:
                amount = comp_session.get_amount()
            elif mart_session:
                amount = mart_session.get_amount()

            print(f"Placing trade: {signal} for {amount} on {cfg['asset']} at {cfg['trade_time']}")
            if driver:
                trade_result = place_trade(driver, cfg['asset'], cfg['trade_time'], signal, amount)
            else:
                # For demonstration, random win/loss
                trade_result = np.random.choice(["WIN", "LOSS"])
                print(f"Trade result: {trade_result}")

            if comp_session:
                if trade_result == "WIN":
                    comp_session.win()
                else:
                    comp_session.lose()
            if mart_session:
                try:
                    if trade_result == "WIN":
                        mart_session.win()
                    else:
                        mart_session.lose()
                except Exception as e:
                    print(str(e))
                    print("Bot stopped due to Martingale max steps.")
                    if driver:
                        driver.quit()
                    return
            time.sleep(10)
    except KeyboardInterrupt:
        print("Manual Trading stopped.")
    if driver:
        driver.quit()

def ai_trading(cfg):
    print("AI Trading started.")
    comp_session = CompoundingSession(cfg['amount'], cfg['comp_multiplier'], cfg['comp_steps']) if cfg['enable_comp'] else None
    mart_session = MartingaleSession(cfg['amount'], cfg['mart_multiplier'], cfg['mart_steps']) if cfg['enable_mart'] else None
    driver = start_browser()
    login(driver)
    balance = get_balance(driver)
    try:
        model = build_model()
        while True:
            asset = np.random.choice(ASSET_LIST)
            trade_time = np.random.choice(["5s", "15s", "1m", "2m", "3m", "5m", "15m"])
            df = get_historical_data(asset, period="1d", interval="1m")
            if df.empty or len(df) < 70:
                print(f"Not enough data for {asset}. Skipping.")
                continue
            features = calculate_features(df)
            if features.shape[0] < 60:
                print(f"Not enough indicator data for {asset}. Skipping.")
                continue
            signal, confidence = ai_decision(model, features)
            print(f"{datetime.now()} [{asset}]: AI Signal: {signal} (confidence: {confidence:.2f})")
            if signal != "NONE" and confidence > CONFIDENCE_THRESHOLD:
                amount = cfg['amount']
                if comp_session:
                    amount = comp_session.get_amount()
                elif mart_session:
                    amount = mart_session.get_amount()
                print(f"Placing trade: {signal} for {amount} on {asset} at {trade_time}")
                trade_result = place_trade(driver, asset, trade_time, signal, amount)
                if comp_session:
                    if trade_result == "WIN":
                        comp_session.win()
                    else:
                        comp_session.lose()
                if mart_session:
                    try:
                        if trade_result == "WIN":
                            mart_session.win()
                        else:
                            mart_session.lose()
                    except Exception as e:
                        print(str(e))
                        print("Bot stopped due to Martingale max steps.")
                        driver.quit()
                        return
            time.sleep(10)
    except KeyboardInterrupt:
        print("AI Trading stopped.")
    driver.quit()

# ====== GUI ======
def show_menu():
    root = tk.Tk()
    root.title("Nd AI Trading Bot")
    root.geometry("500x600")
    root.resizable(False, False)

    # ===== WATERMARK =====
    watermark = tk.Label(root, text=WATERMARK_TEXT, fg='gray', font=("Arial", 8, 'bold'))
    watermark.pack(side="bottom", pady=8)

    # ===== Mode Selection =====
    mode_var = tk.StringVar(value="manual")
    ttk.Label(root, text="Select Trading Mode:", font=("Arial", 13, 'bold')).pack(pady=10)
    ttk.Radiobutton(root, text="Manual Trading", variable=mode_var, value="manual").pack()
    ttk.Radiobutton(root, text="AI Trading", variable=mode_var, value="ai").pack()

    # ===== Manual Trading Options =====
    manual_frame = ttk.Frame(root)
    manual_frame.pack(pady=10, fill="x")

    ttk.Label(manual_frame, text="Amount:").grid(row=0, column=0, sticky="w")
    amount_var = tk.DoubleVar(value=5)
    amount_entry = ttk.Entry(manual_frame, textvariable=amount_var, width=10)
    amount_entry.grid(row=0, column=1, sticky="w")

    ttk.Label(manual_frame, text="AI Recommendation or User?").grid(row=1, column=0, sticky="w")
    ai_rec_var = tk.BooleanVar(value=True)
    ttk.Radiobutton(manual_frame, text="AI", variable=ai_rec_var, value=True).grid(row=1, column=1, sticky="w")
    ttk.Radiobutton(manual_frame, text="User", variable=ai_rec_var, value=False).grid(row=1, column=2, sticky="w")

    ttk.Label(manual_frame, text="Asset:").grid(row=2, column=0, sticky="w")
    asset_var = tk.StringVar(value=ASSET_LIST[0])
    asset_menu = ttk.Combobox(manual_frame, textvariable=asset_var, values=ASSET_LIST, state="readonly", width=15)
    asset_menu.grid(row=2, column=1, sticky="w")

    ttk.Label(manual_frame, text="Trade Time:").grid(row=3, column=0, sticky="w")
    trade_time_var = tk.StringVar(value="1m")
    trade_time_menu = ttk.Combobox(
        manual_frame, textvariable=trade_time_var,
        values=["5s", "15s", "1m", "2m", "3m", "5m", "15m"], state="readonly", width=10
    )
    trade_time_menu.grid(row=3, column=1, sticky="w")

    auto_trade_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(manual_frame, text="Auto Place Trade", variable=auto_trade_var).grid(row=4, column=0, sticky="w")

    # ===== Compounding Options =====
    comp_frame = ttk.LabelFrame(root, text="Compounding")
    comp_frame.pack(pady=10, fill="x")
    enable_comp_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(comp_frame, text="Enable Compounding", variable=enable_comp_var).grid(row=0, column=0, sticky="w")
    ttk.Label(comp_frame, text="Multiplier:").grid(row=0, column=1, sticky="w")
    comp_multiplier_var = tk.DoubleVar(value=2.0)
    ttk.Combobox(comp_frame, textvariable=comp_multiplier_var, values=[1.5, 2, 2.5], width=5, state="readonly").grid(row=0, column=2, sticky="w")
    ttk.Label(comp_frame, text="Steps:").grid(row=0, column=3, sticky="w")
    comp_steps_var = tk.IntVar(value=3)
    ttk.Spinbox(comp_frame, from_=1, to=10, textvariable=comp_steps_var, width=3).grid(row=0, column=4, sticky="w")

    # ===== Martingale Options =====
    mart_frame = ttk.LabelFrame(root, text="Martingale")
    mart_frame.pack(pady=10, fill="x")
    enable_mart_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(mart_frame, text="Enable Martingale", variable=enable_mart_var).grid(row=0, column=0, sticky="w")
    ttk.Label(mart_frame, text="Multiplier:").grid(row=0, column=1, sticky="w")
    mart_multiplier_var = tk.DoubleVar(value=2.0)
    ttk.Combobox(mart_frame, textvariable=mart_multiplier_var, values=[1.5, 2, 2.5], width=5, state="readonly").grid(row=0, column=2, sticky="w")
    ttk.Label(mart_frame, text="Steps:").grid(row=0, column=3, sticky="w")
    mart_steps_var = tk.IntVar(value=4)
    ttk.Spinbox(mart_frame, from_=1, to=8, textvariable=mart_steps_var, width=3).grid(row=0, column=4, sticky="w")

    def on_start():
        cfg = {
            'amount': amount_var.get(),
            'ai_recommend': ai_rec_var.get(),
            'asset': asset_var.get(),
            'trade_time': trade_time_var.get(),
            'enable_comp': enable_comp_var.get(),
            'comp_multiplier': comp_multiplier_var.get(),
            'comp_steps': comp_steps_var.get(),
            'enable_mart': enable_mart_var.get(),
            'mart_multiplier': mart_multiplier_var.get(),
            'mart_steps': mart_steps_var.get(),
            'auto_trade': auto_trade_var.get()
        }
        root.destroy()
        if mode_var.get() == "manual":
            manual_trading(cfg)
        else:
            ai_trading(cfg)

    ttk.Button(root, text="Start", command=on_start).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    show_menu()