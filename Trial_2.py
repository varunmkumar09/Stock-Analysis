import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


pd.options.mode.chained_assignment = None  # default='warn'

# Define the stock symbol
symbol = 'COFORGE.NS'  # Example: State Bank of India

a = datetime.datetime.today()
Today = a.strftime('%Y-%m-%d')  # Fetch the historical data
data = yf.download(symbol, start='2022-01-01', end=Today)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date
data = data[data['Volume'] != 0]

# Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)
sma_window = 10  # Define the SMA window
ema_window = 10  # Define the EMA window

data.loc[:, 'SMA'] = data.loc[:, 'Close'].rolling(window=sma_window).mean()
data.loc[:, 'EMA'] = data['Close'].ewm(span=ema_window, adjust=False).mean()

# Identify trend direction based on moving averages
data['SMA_Upward'] = np.where(data['SMA'] > data['SMA'].shift(), True, False)
data['EMA_Upward'] = np.where(data['EMA'] > data['EMA'].shift(), True, False)

# Combine the trends from SMA and EMA into a single column 'Trend'
data['Trend'] = np.where((data['SMA_Upward'] & data['EMA_Upward']), 'Uptrend',
                         np.where((~data['SMA_Upward'] & ~data['EMA_Upward']), 'Downtrend', 'Mixed or Neutral'))

# Convert boolean columns to boolean data type
data['SMA_Upward'] = data['SMA_Upward'].astype(bool)
data['EMA_Upward'] = data['EMA_Upward'].astype(bool)


def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame of historical prices.

    Parameters:
        data (pandas.DataFrame): DataFrame containing historical price data with a 'Close' column.
        window (int): Window size for calculating the RSI. Default is 14 periods.

    Returns:
        pandas.Series: A Series containing the RSI values.
    """
    delta = data['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# You can calculate the RSI as follows:
data['RSI'] = calculate_rsi(data)

# Add a new column 'Signal' to indicate overbought, oversold, or neutral zone
data['Signal'] = 'Neutral'
data.loc[data['RSI'] > 70, 'Signal'] = 'Overbought'
data.loc[data['RSI'] < 30, 'Signal'] = 'Oversold'


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

    return macd_line, signal_line


def generate_signals_MACD(data):
    signals = []
    positions = 0

    for i in range(len(data)):
        if data['MACD'][i] > data['Signal Line'][i] and data['MACD'][i - 1] <= data['Signal Line'][i - 1]:
            signals.append(1)  # Buy signal
            positions = 1
        elif data['MACD'][i] < data['Signal Line'][i] and data['MACD'][i - 1] >= data['Signal Line'][i - 1]:
            signals.append(-1)  # Sell signal
            positions = -1
        else:
            signals.append(positions)

    return signals


# Calculate MACD and Signal Line
macd_line, signal_line = calculate_macd(data)

# Add MACD and Signal Line to DataFrame
data['MACD'] = macd_line
data['Signal Line'] = signal_line

# Reset the index to make it continuous
data.reset_index(inplace=True)

# Generate buy and sell signals based on MACD crossovers
data['MACD Signal'] = generate_signals_MACD(data)


def calculate_bollinger_bands(data, window=10):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    return upper_band, lower_band


def generate_signals_Bollinger(data):
    signals = []
    positions = 0

    for i in range(len(data)):
        if data['Close'][i] < data['Lower Band'][i] and data['Close'][i - 1] > data['Lower Band'][i - 1]:
            signals.append(1)  # Buy signal
            positions = 1
        elif data['Close'][i] > data['Upper Band'][i] and data['Close'][i - 1] < data['Upper Band'][i - 1]:
            signals.append(-1)  # Sell signal
            positions = -1
        else:
            signals.append(positions)

    return signals


# Calculate Bollinger Bands
upper_band, lower_band = calculate_bollinger_bands(data, window=10)

data['Upper Band'] = upper_band
data['Lower Band'] = lower_band

# Generate buy and sell signals based on Bollinger Bands
data['Bollinger Signal'] = generate_signals_Bollinger(data)


# Stochastic Oscillator
def calculate_stochastic_oscillator(data, window=14, smooth_window=3):
    # Calculate the lowest price over the window period
    data['Lowest Low'] = data['Low'].rolling(window=window).min()

    # Calculate the highest price over the window period
    data['Highest High'] = data['High'].rolling(window=window).max()

    # Calculate %K
    data['%K'] = 100 * (data['Close'] - data['Lowest Low']) / (data['Highest High'] - data['Lowest Low'])

    # Calculate %D (smoothing %K)
    data['%D'] = data['%K'].rolling(window=smooth_window).mean()

    return data['%K'], data['%D']


# Calculate Stochastic Oscillator (%K and %D)
k_line, d_line = calculate_stochastic_oscillator(data)


def calculate_fibonacci_retracement(data, swing_high, swing_low):
    # Calculate price range between the swing high and swing low
    price_range = swing_high - swing_low

    # Fibonacci retracement levels
    retracement_23_6 = swing_high - (0.236 * price_range)
    retracement_38_2 = swing_high - (0.382 * price_range)
    retracement_50_0 = swing_high - (0.5 * price_range)
    retracement_61_8 = swing_high - (0.618 * price_range)
    retracement_78_6 = swing_high - (0.786 * price_range)

    return retracement_23_6, retracement_38_2, retracement_50_0, retracement_61_8, retracement_78_6


# Identify significant swing high and swing low points
swing_high = data['High'].max()
swing_low = data['Low'].min()

# Calculate Fibonacci retracement levels
fib_23_6, fib_38_2, fib_50_0, fib_61_8, fib_78_6 = calculate_fibonacci_retracement(data, swing_high, swing_low)


def analyze_volume(data):
    # Calculate the moving average of volume (you can adjust the window as needed)
    data['Volume MA'] = data['Volume'].rolling(window=10).mean()

    # Volume confirmation: Check if volume increases with price trends
    data['Volume Trend Confirmation'] = data['Volume'] > data['Volume MA']

    # Volume divergence: Check if volume decreases with price trends
    data['Volume Trend Divergence'] = data['Volume'] < data['Volume MA']

    # Volume clusters: Identify high volume areas
    data['Volume Cluster'] = data['Volume'] > data['Volume'].quantile(0.9)

    return data


def generate_signals(data):
    signals = []

    for i in range(1, len(data)):
        if data['Volume Trend Confirmation'][i] and not data['Volume Trend Confirmation'][i - 1]:
            signals.append(1)  # Buy signal
        elif data['Volume Trend Divergence'][i] and not data['Volume Trend Divergence'][i - 1]:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # No signal

    signals = [0] + signals  # Add initial 0 for consistency with DataFrame length
    return signals


# Analyze volume
data = analyze_volume(data)

data['Volume Analysis Signal'] = generate_signals(data)


def calculate_volume_breakout(data, threshold_factor=2.0):
    data['Volume MA'] = data['Volume'].rolling(window=10).mean()
    data['Volume Std'] = data['Volume'].rolling(window=10).std()

    # Calculate dynamic threshold based on threshold factor and volume standard deviation
    data['Threshold'] = data['Volume MA'] + threshold_factor * data['Volume Std']

    # Check if volume is greater than the threshold
    data['Volume Breakout'] = data['Volume'] > data['Threshold']

    return data

# Calculate dynamic threshold based on historical volume volatility
data = calculate_volume_breakout(data, threshold_factor=2.0)

def calculate_pivot_points(data):
    # Calculate Pivot Point (PP)
    data['Pivot Point (PP)'] = (data['High'] + data['Low'] + data['Close']) / 3

    # Calculate Support and Resistance levels
    data['Support 1 (S1)'] = 2 * data['Pivot Point (PP)'] - data['High']
    data['Resistance 1 (R1)'] = 2 * data['Pivot Point (PP)'] - data['Low']

    data['Support 2 (S2)'] = data['Pivot Point (PP)'] - (data['High'] - data['Low'])
    data['Resistance 2 (R2)'] = data['Pivot Point (PP)'] + (data['High'] - data['Low'])

    data['Support 3 (S3)'] = data['Low'] - 2 * (data['High'] - data['Pivot Point (PP)'])
    data['Resistance 3 (R3)'] = data['High'] + 2 * (data['Pivot Point (PP)'] - data['Low'])

    return data

# Calculate Pivot Points
data = calculate_pivot_points(data)

# Print the Pivot Points
#print(data[['Pivot Point (PP)', 'Support 1 (S1)', 'Resistance 1 (R1)', 'Support 2 (S2)', 'Resistance 2 (R2)', 'Support 3 (S3)', 'Resistance 3 (R3)']].tail())

# Analyzing EMA and SMA significance
last_row = data.iloc[-1]
if last_row['SMA_Upward'] and last_row['EMA_Upward']:
    print("The stock is in an Uptrend based on both SMA and EMA.")
elif not last_row['SMA_Upward'] and not last_row['EMA_Upward']:
    print("The stock is in a Downtrend based on both SMA and EMA.")
else:
    print("The stock is in a Mixed or Neutral trend based on SMA and EMA.")

# Analyzing RSI significance
last_row = data.iloc[-1]
if last_row['RSI'] > 70:
    print("The stock is overbought based on RSI.")
elif last_row['RSI'] < 30:
    print("The stock is oversold based on RSI.")
else:
    print("The stock is in a neutral zone based on RSI.")


# Analyzing Volume Analysis
data["Signal From Volume Analysis"] = 'Neutral'
for i in range(len(data)):
    if data['Volume Analysis Signal'][i] == 1:
        data["Signal From Volume Analysis"][i] = 'Buy'
    elif data['Volume Analysis Signal'][i] == -1:
        data["Signal From Volume Analysis"][i] = 'Sell'

print(data)

"""

# Create a new column 'Actual Movement' to track the actual price movement
data['Actual Movement'] = np.where(data['Close'] > data['Close'].shift(), 'Up', np.where(data['Close'] < data['Close'].shift(), 'Down', 'Neutral'))

# Create columns to track the accuracy of buy and sell signals
data['Buy Accuracy'] = 0
data['Sell Accuracy'] = 0

# Loop through the DataFrame to check the accuracy of buy and sell signals
for i in range(1, len(data)):
    if data['MACD Signal'][i] == 1 and data['Actual Movement'][i] == 'Up':
        data.at[i, 'Buy Accuracy'] = 1
    elif data['MACD Signal'][i] == -1 and data['Actual Movement'][i] == 'Down':
        data.at[i, 'Sell Accuracy'] = 1

    if data['Bollinger Signal'][i] == 1 and data['Actual Movement'][i] == 'Up':
        data.at[i, 'Buy Accuracy'] = 1
    elif data['Bollinger Signal'][i] == -1 and data['Actual Movement'][i] == 'Down':
        data.at[i, 'Sell Accuracy'] = 1

    if data['Volume Analysis Signal'][i] == 1 and data['Actual Movement'][i] == 'Up':
        data.at[i, 'Buy Accuracy'] = 1
    elif data['Volume Analysis Signal'][i] == -1 and data['Actual Movement'][i] == 'Down':
        data.at[i, 'Sell Accuracy'] = 1
        

# Calculate the accuracy percentage for buy and sell signals
total_buy_signals = data['Buy Accuracy'].sum()
total_sell_signals = data['Sell Accuracy'].sum()
total_buy_accuracy = total_buy_signals / len(data) * 100
total_sell_accuracy = total_sell_signals / len(data) * 100

print(f"Total Buy Signals: {total_buy_signals}")
print(f"Total Sell Signals: {total_sell_signals}")
print(f"Buy Accuracy: {total_buy_accuracy:.2f}%")
print(f"Sell Accuracy: {total_sell_accuracy:.2f}%")

print('\n')


# Prepare data for machine learning
# Select relevant features
features = ['SMA_Upward', 'EMA_Upward', 'RSI', 'MACD Signal', 'Bollinger Signal', 'Volume Analysis Signal']

# Create target variable (labels)
data['Signal_Label'] = 'Hold'  # Initialize all signals as "Hold"

# Assign buy and sell labels based on the generated signals
data.loc[data['Volume Analysis Signal'] == 1, 'Signal_Label'] = 'Buy'
data.loc[data['Volume Analysis Signal'] == -1, 'Signal_Label'] = 'Sell'

# Split data into training and test sets
X = data[features]
y = data['Signal_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))


# Plot for data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Historical Stock Prices')
plt.xticks(rotation=45)
plt.show()

# Plot for SMA and EMA
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Price')
plt.plot(data['Date'], data['SMA'], label='SMA')
plt.plot(data['Date'], data['EMA'], label='EMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price with SMA and EMA')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Plotting the RSI
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['RSI'], label='RSI', color='blue')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.title('Relative Strength Index (RSI)')
plt.legend()
plt.show()

# Plotting the MACD and Signal Line
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['MACD'], label='MACD', color='blue')
plt.plot(data['Date'], data['Signal Line'], label='Signal Line', color='red')
plt.axhline(0, color='gray', linestyle='--', label='Zero Line')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.title('MACD and Signal Line')
plt.legend()
plt.show()

# Plotting the Bollinger Bands and buy/sell signals
data.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Price', color='blue')
plt.plot(data.index, data['Upper Band'], label='Upper Band', color='red')
plt.plot(data.index, data['Lower Band'], label='Lower Band', color='green')
plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], alpha=0.2, color='gray')

buy_signals = data[data['Bollinger Signal'] == 1]
sell_signals = data[data['Bollinger Signal'] == -1]
plt.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Buy Signal')
plt.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bollinger Bands with Buy/Sell Signals')
plt.legend()
plt.show()

# Convert 'Date' index back to a column
data.reset_index(inplace=True)

# Plotting %K, %D, and buy/sell signals
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['%K'], label='%K', color='blue')
plt.plot(data['Date'], data['%D'], label='%D', color='red')
plt.axhline(80, color='gray', linestyle='--', label='Overbought (80)')
plt.axhline(20, color='gray', linestyle='--', label='Oversold (20)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Stochastic Oscillator with Buy/Sell Signals')
plt.legend()
plt.show()

# Plotting the price chart with Fibonacci retracement levels
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Price', color='blue')

plt.axhline(swing_high, color='red', linestyle='--', label='Swing High')
plt.axhline(swing_low, color='green', linestyle='--', label='Swing Low')

plt.axhline(fib_23_6, color='orange', linestyle='--', label='23.6%')
plt.axhline(fib_38_2, color='purple', linestyle='--', label='38.2%')
plt.axhline(fib_50_0, color='cyan', linestyle='--', label='50.0%')
plt.axhline(fib_61_8, color='magenta', linestyle='--', label='61.8%')
plt.axhline(fib_78_6, color='brown', linestyle='--', label='78.6%')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Fibonacci Retracement Levels')
plt.legend()
plt.show()

# Plotting volume analysis
plt.figure(figsize=(12, 8))

# Price chart
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Close'], label='Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Chart')
plt.legend()

# Volume chart with trend confirmation, divergence, and clusters
plt.subplot(2, 1, 2)
plt.bar(data['Date'], data['Volume'], label='Volume', color='gray')
plt.plot(data['Date'], data['Volume MA'], label='Volume MA', color='red', linestyle='dashed')
plt.scatter(data['Date'][data['Volume Trend Confirmation']], data['Volume'][data['Volume Trend Confirmation']],
            marker='^', color='g', label='Volume Trend Confirmation')
plt.scatter(data['Date'][data['Volume Trend Divergence']], data['Volume'][data['Volume Trend Divergence']], marker='v',
            color='r', label='Volume Trend Divergence')
plt.bar(data['Date'][data['Volume Cluster']], data['Volume'][data['Volume Cluster']], color='orange',
        label='Volume Cluster')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume Analysis')
plt.legend()

plt.tight_layout()
plt.show()


# Plotting the Pivot Points and Levels on a price chart
plt.figure(figsize=(12, 6))

# Price chart
plt.plot(data['Date'], data['Close'], label='Close', color='blue')

# Pivot Point (PP) line
plt.axhline(y=data['Pivot Point (PP)'].iloc[-1], color='purple', linestyle='--', label='Pivot Point (PP)')

# Support and Resistance levels lines
plt.axhline(y=data['Support 1 (S1)'].iloc[-1], color='green', linestyle='--', label='Support 1 (S1)')
plt.axhline(y=data['Resistance 1 (R1)'].iloc[-1], color='red', linestyle='--', label='Resistance 1 (R1)')

plt.axhline(y=data['Support 2 (S2)'].iloc[-1], color='green', linestyle='--', label='Support 2 (S2)')
plt.axhline(y=data['Resistance 2 (R2)'].iloc[-1], color='red', linestyle='--', label='Resistance 2 (R2)')

plt.axhline(y=data['Support 3 (S3)'].iloc[-1], color='green', linestyle='--', label='Support 3 (S3)')
plt.axhline(y=data['Resistance 3 (R3)'].iloc[-1], color='red', linestyle='--', label='Resistance 3 (R3)')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Standard Pivot Points and Support/Resistance Levels')
plt.legend()
plt.show()
"""