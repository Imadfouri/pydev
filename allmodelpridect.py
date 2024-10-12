import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime


# Get today's date in a specific format
today_formatted = datetime.today().strftime('%Y-%m-%d')  # Example format: 'YYYY-MM-DD'

# List of top 20 cryptocurrencies and all major forex pairs
crypto_pairs = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 
    'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'SHIB-USD', 'AVAX-USD', 'TRX-USD', 
    'LTC-USD', 'UNI-USD', 'ATOM-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 
    'VET-USD', 'ICP-USD'
]

forex_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 
    'NZDUSD=X', 'EURJPY=X', 'EURGBP=X', 'GBPJPY=X', 'EURAUD=X', 'EURCAD=X', 
    'AUDJPY=X', 'AUDCAD=X', 'CHFJPY=X', 'GBPCHF=X'
]

gold_oile = ['GC=F','CL=F','^GSPC']



# Fetch data
@st.cache
def load_data(symbol, start_date='2015-01-01', end_date=today_formatted):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# App Layout
st.title("Crypto & Forex Price Prediction with Deep Learning, Linear Regression, and Random Forest")

# Select Market: Crypto or Forex
category = st.selectbox("Select Market", ["Crypto", "Forex","gold_oile"])

if category == "Crypto":
    # Select from top 20 cryptocurrencies
    symbol = st.selectbox("Select Cryptocurrency", crypto_pairs)
elif category == "Forex":
    # Select from all available forex pairs
    symbol = st.selectbox("Select Forex Pair", forex_pairs)
elif category == "gold_oile":
    symbol = st.selectbox("Select gold_oile Pair", gold_oile)

# Fetch Data
data = load_data(symbol)

# Manual Data Input
st.write("## Manual Data Input")
date = st.date_input("Date")
close_price = st.number_input("Close Price", min_value=0.0, step=1.0)

if st.button("Add Data Point"):
    new_data = pd.DataFrame({
        'Date': [date],
        'Close': [close_price]
    })
    new_data.set_index('Date', inplace=True)
    data = pd.concat([data, new_data])
    data.sort_index(inplace=True)

# Display Data
st.write(f"## {symbol} Historical Data")
st.write(data.head())

# Plot Closing Price
st.write("## Closing Price")
fig, ax = plt.subplots()
ax.plot(data['Close'])
ax.set_title(f'{symbol} Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
st.pyplot(fig)

# Moving Averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

def weighted_moving_average(values, window):
    weights = np.arange(1, window + 1)
    return values.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

data['WMA50'] = weighted_moving_average(data['Close'], 50)
data['WMA200'] = weighted_moving_average(data['Close'], 200)

# Plot Moving Averages
st.write("## Moving Averages")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close')
ax.plot(data['SMA50'], label='50-day SMA')
ax.plot(data['SMA200'], label='200-day SMA')
ax.plot(data['EMA50'], label='50-day EMA')
ax.plot(data['EMA200'], label='200-day EMA')
ax.plot(data['WMA50'], label='50-day WMA')
ax.plot(data['WMA200'], label='200-day WMA')
ax.set_title(f'{symbol} Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# Feature Creation
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std()
data = data.dropna()

# Define Target Variable
data['Target'] = data['Return'].shift(-1)
data = data.dropna()

# Scale the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[['Return', 'Volatility']])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['Target'], test_size=0.2, random_state=42)

# Deep Learning Model
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train Deep Learning Model
model = build_model()
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Deep Learning Predictions
dl_predictions = model.predict(X_test)
dl_mse = mean_squared_error(y_test, dl_predictions)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Display Evaluations
st.write("## Model Evaluations")
st.write(f"Deep Learning MSE: {dl_mse}")
st.write(f"Linear Regression MSE: {lr_mse}")
st.write(f"Random Forest MSE: {rf_mse}")

# Plot Predictions vs Actual for each model
st.write("## Model Predictions")
fig, ax = plt.subplots()
ax.plot(y_test.values, label='Actual')
ax.plot(dl_predictions, label='Deep Learning Predictions')
ax.plot(lr_predictions, label='Linear Regression Predictions')
ax.plot(rf_predictions, label='Random Forest Predictions')
ax.set_title('Model Predictions vs Actual')
ax.set_xlabel('Index')
ax.set_ylabel('Return')
ax.legend()
st.pyplot(fig)

# Predict Next Day Return
latest_data = scaler.transform(data[['Return', 'Volatility']].iloc[-1].values.reshape(1, -1))

dl_next_day_prediction = model.predict(latest_data)[0][0]
lr_next_day_prediction = lr_model.predict(latest_data)[0]
rf_next_day_prediction = rf_model.predict(latest_data)[0]

last_close_price = data['Close'].iloc[-1]
dl_predicted_close = last_close_price * (1 + dl_next_day_prediction)
lr_predicted_close = last_close_price * (1 + lr_next_day_prediction)
rf_predicted_close = last_close_price * (1 + rf_next_day_prediction)

st.write("## Next Day Predictions")
st.write(f"Deep Learning Predicted Next Day Close Price: ${dl_predicted_close:.2f}")
st.write(f"Linear Regression Predicted Next Day Close Price: ${lr_predicted_close:.2f}")
st.write(f"Random Forest Predicted Next Day Close Price: ${rf_predicted_close:.2f}")

"========================== News Sentiment Analysis ================="
import plotly.express as px
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize News API (replace 'your_api_key' with your NewsAPI key)
newsapi = NewsApiClient(api_key='7c49009974314836ad80c969e3966dbb')

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to fetch stock/forex/crypto data
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return hist['Close'].iloc[-1]  # Return last close price

# Function to fetch news
def fetch_news(query):
    news = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
    articles = news['articles'][:5]  # Top 5 news
    return [(article['title'], article['url'], article['description']) for article in articles]

# Function to analyze sentiment
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score

# Streamlit App
st.title("Market Risk Prediction & News Sentiment Analysis")

# Sidebar for market selection
st.sidebar.header("Select Market")
option = st.sidebar.selectbox("Choose a market", ("Cryptocurrency", "Forex", "Stocks"))

# Example data and news topics
if option == "Cryptocurrency":
    cryptos = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
    data = {name: fetch_data(ticker) for name, ticker in cryptos.items()}
    news_topic = "cryptocurrency"
elif option == "Forex":
    forex_pairs = {"EURUSD": "EURUSD=X", "USDJPY": "JPY=X"}
    data = {pair: fetch_data(ticker) for pair, ticker in forex_pairs.items()}
    news_topic = "forex"
elif option == "Stocks":
    stocks = {"Apple": "AAPL", "Tesla": "TSLA", "Amazon": "AMZN"}
    data = {name: fetch_data(ticker) for name, ticker in stocks.items()}
    news_topic = "stock market"

# Fetch news and sentiment analysis
st.subheader(f"{option} News & Sentiment")
news = fetch_news(news_topic)

# Analyze sentiment of news and classify as positive, neutral, or negative
sentiment_scores = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
for title, url, description in news:
    st.markdown(f"[{title}]({url})")
    sentiment = analyze_sentiment(description)
    st.write(f"Sentiment Score: {sentiment}")
    
    # Classify sentiment
    if sentiment['compound'] >= 0.05:
        sentiment_scores['Positive'] += 1
    elif sentiment['compound'] <= -0.05:
        sentiment_scores['Negative'] += 1
    else:
        sentiment_scores['Neutral'] += 1

# Plot sentiment-based risk as pie chart
st.subheader("Risk Prediction Based on Sentiment")
fig = px.pie(values=list(sentiment_scores.values()), names=list(sentiment_scores.keys()), 
             title=f"{option} Risk Prediction")
st.plotly_chart(fig)
