import streamlit as st
from datetime import date


import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly as px
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from PIL import Image



START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stockflix : Stock Forecast App 💎')



st.write('This application allows you to generate stock price predictions for the most important stocks.')


stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA','AMZN','BTC-USD','ETH-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years* 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.subheader('1.Data Loading 🏋️')
	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('2.Data Analysis 📊')
st.markdown('Once the data is loaded we move on to the exploratory analysis.')

st.subheader('Raw data')
st.markdown('Below we have the last 5 observations of the stock')
st.write(data.tail())

st.subheader('Descriptive Statistics')
st.markdown('You can observe the maximums, minimums, standard deviation, average price.')
st.write(data.describe())

# Plot raw data
st.subheader(' Line Plot ')

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Candle Plot
st.subheader(' Candlestick Plot: Price Evolution')

def plot_candle_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))
    fig.update_layout(
    title='Stock Share Price Evolution',
    yaxis_title='Stock Price (USD per Shares)')
    st.plotly_chart(fig)

plot_candle_data()


    
# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods=period,freq = 'D')
forecast = m.predict(future)

# Show and plot forecast
st.subheader('3.Forecast data 🔮')
st.write("The model is trained with the data and generates predictions.")
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components 📚")
st.write("We load the model components.")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.markdown('The first graph shows information about the trend.')
st.markdown(' The second chart shows information about the weekly trend.')
st.markdown('The last graph provides us with information about the annual holding.')

st.subheader('ChangePoints Plot 🔱')

fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
st.write(fig3)


                                                 
st.title('Authors')
st.subheader('Saharsh Saxena : purplerain007')
