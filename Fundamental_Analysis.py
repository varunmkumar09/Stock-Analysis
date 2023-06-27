# Importing Required Libraries
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries

# Alpha Vantage API Key
api_key = 'IO36W7YZDX42OL9N'

# Stock Ticker
ticker = 'GICRE.BSE'

# API Endpoint for Financials
url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'

# Fetching Data
data = requests.get(url).json()

# Extracting Financial Data
price = data['Global Quote']['05. price']
# Calculating P/E Ratio
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
data = requests.get(url).json()
earnings = data['Time Series (Daily)']['4. close']['earnings per share']

stock_data = pd.DataFrame({'price': price, 'earnings': earnings}, index=[0])
stock_data["P/E Ratio"] = stock_data["price"] / stock_data["earnings"]

print(stock_data[["P/E Ratio"]])



# Creating Dataframe
stock_data = pd.DataFrame({'price': price, 'earnings': earnings}, index=[0])

# Plotting the Data
stock_data.plot(kind='bar', y='price', x='earnings')
plt.show()


# Calculating P/E Ratio
stock_data["P/E Ratio"] = stock_data["price"] / stock_data["earnings"]

#Calculating Price-to-book Ratio
url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}'
data = requests.get(url).json()
book_value = data['annualReports'][0]['totalStockholderEquity']
stock_data["Price-to-book Ratio"] = stock_data["price"] / book_value

#Calculating Dividend yield
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
data = requests.get(url).json()
dividend = data['Time Series (Daily)']['4. close']['dividend amount']
stock_data["Dividend Yield"] = dividend / stock_data["price"]

# Displaying the Ratios
print(stock_data[["P/E Ratio","Price-to-book Ratio","Dividend Yield"]])




# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(stock_data[["P/E Ratio"]], stock_data["price"], test_size=0.2)

# Model Training
reg = LinearRegression().fit(X_train, y_train)

# Model Evaluation
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
