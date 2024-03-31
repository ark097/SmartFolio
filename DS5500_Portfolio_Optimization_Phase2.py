#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from fredapi import Fred
from scipy.optimize import minimize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px

# In[10]:


def get_stock_data(ticker, start_date, end_date):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# In[84]:


def plot_chart(data, ticker):
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


# In[85]:


def forecast_stock_price(data, ticker):
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

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights*252)

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def get_risk_free_rate():
    fred = Fred(api_key='37d707ea29d05517c0f3f400b23644bb')
    treasury_rate_10y = fred.get_series_latest_release('GS10')/100
    risk_free_rate = treasury_rate_10y.iloc[-1]
    return risk_free_rate

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def plot_portfolio_weights_chart(weights, tickers):
    filtered_weights = [w for w in weights if w > 0.01]
    filtered_tickers = [tickers[i] for i, w in enumerate(weights) if w > 0.01]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.pie(filtered_weights, labels=filtered_tickers, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_xlabel('Assets')
    ax.set_ylabel('Optimal Weights')
    ax.set_title('Optimal Portfolio Weights (CG)')
    return fig

def build_portfolio(tickers, risk_free_rate):
    adj_close_df = pd.DataFrame()
    end = datetime.today()
    start = end - timedelta(days = 3*365)

    for ticker in tickers:
        data = yf.download(ticker, start = start,end = end)
        adj_close_df[ticker] = data['Adj Close']
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.5) for _ in range(len(tickers))]
    initial_weights = np.array([1/len(tickers)]*len(tickers))
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns.dropna(inplace=True)
    cov_matrix = log_returns.cov() * 252
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimized_results.x
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    return optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio


def main():
    risk_free_rate = get_risk_free_rate()
    ticker_file = open("all_tickers.txt", "r") 
    all_tickers = ticker_file.read().split('\n')
    st.title('Portfolio Optimization Tool')

    st.sidebar.title('Navigate')
    page = st.sidebar.selectbox('Page', ["Explore Stocks", "Build Your Portfolio"])

    if page == "Explore Stocks":
        st.subheader('View Stock Data')
        
        ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):').upper()
        
        start_date = st.date_input('Select Start Date:')
        end_date = st.date_input('Select End Date:')
    
        if ticker:
            data = get_stock_data(ticker, start_date, end_date)
            if not data.empty:
                
                if st.button('View Chart'):
                    plot_chart(data, ticker)
                    
                st.subheader('Make a Forecast')
                if st.button('Forecast'):
                    forecast_stock_price(data, ticker)
            else:
                st.warning('No data found for the selected stock ticker and timeframe.')

    if page == "Build Your Portfolio":
        tickers = st.multiselect('Select Stock Tickers', all_tickers)
        submit_button = st.button('Submit')
        if submit_button and tickers:
            weights, returns, volatility, sharpe_ratio = build_portfolio(tickers, risk_free_rate)

            st.pyplot(plot_portfolio_weights_chart(weights, tickers))
            data = {'Tickers': tickers, 'Weights': weights}
            df = pd.DataFrame(data)
            st.table(df)
            st.write(f'Expected Returns: {round(returns*100,2)}%')
            st.write(f'Expected Volatility: {round(volatility*100,2)}%')
            st.write(f'Sharpe Ratio: {round(sharpe_ratio, 4)}')
# In[83]:


if __name__ == "__main__":
    main()


# In[ ]:




