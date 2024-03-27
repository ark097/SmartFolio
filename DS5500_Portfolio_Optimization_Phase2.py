#!/usr/bin/env python
# coding: utf-8

# In[47]:


import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go


# In[10]:


def get_stock_data(ticker, start_date, end_date):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# In[68]:


def plot_chart(data):
    data = data.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], mode='lines', name=f'{ticker} Stock Price'))

    fig.update_layout(
        title=f'{ticker} Stock Price',
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            rangeslider=dict(
                visible=False
            ),
            type="date"
        )
    )
    st.plotly_chart(fig)


# In[79]:


def forecast_stock_price(data):
    if len(data) >= 251:
        data = data.reset_index()
        df = data[['Date', 'Adj Close']]
        df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)
        model = Prophet(daily_seasonality = True)
        model.fit(df)
        future = model.make_future_dataframe(periods=120)
        pred = model.predict(future)
        fig = plot_plotly(model, pred)
        fig.update_layout(
            title=f'{ticker} Price Forecast',
            xaxis_title="Date",
            yaxis_title="Price"
        )
        st.plotly_chart(fig)
    else:
        st.warning("Not enough historical data available for forecasting. (Select at least twelve months of data to forecast)")


# In[70]:


st.title('Portfolio Optimization Tool')

ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):')
start_date = st.date_input('Select Start Date:')
end_date = st.date_input('Select End Date:')

if ticker:
    data = get_stock_data(ticker, start_date, end_date)
    if not data.empty:
        
        if st.button('View Chart'):
            plot_chart(data)

        if st.button('Forecast'):
            forecast_stock_price(data)
    else:
        st.warning('No data found for the selected stock ticker and timeframe.')


# In[ ]:




