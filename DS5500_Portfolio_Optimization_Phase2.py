#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## Stock forecasting

# In[2]:


def get_stock_data(ticker, start_date, end_date):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# In[3]:


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


# In[4]:


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


# ## Markowitz model

# In[5]:


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


# In[6]:


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


# In[7]:


def build_portfolio(tickers, risk_free_rate, start_date, end_date):
    adj_close_df = pd.DataFrame()
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')

    for ticker in tickers:
        data = yf.download(ticker, start = start,end = end)
        adj_close_df[ticker] = data['Adj Close']
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(tickers))]
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


# ## Black-Litterman model

# In[8]:


from pypfopt import risk_models, black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import EfficientFrontier, objective_functions

def black_litterman_portfolio(symbols, viewdict, start_date, end_date):
    
    # Download market prices for SPY ETF
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    market_prices = yf.download("SPY", start=start, end=end)["Adj Close"]

    # Get market capitalization for each stock in the portfolio
    mcaps = {}
    for t in symbols:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]

    #Download stock data for the user-requested tickers
    portfolio = yf.download(symbols, start=start, end=end)['Adj Close']
    
    # Calculate Sigma and Delta to get implied market returns
    S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    uncertainty_matrix = {}
    for ticker, view in viewdict.items():
    # Calculate the standard deviation assuming a 68% confidence interval
        std_dev = abs(view) / 0.6745
        uncertainty_matrix[ticker] = std_dev

    omega = np.diag(list(uncertainty_matrix.values()))

    # Instantiate the Black-Litterman model
    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega=omega)

    # Posterior estimate of returns
    ret_bl = bl.bl_returns()

    # Create a DataFrame to display results
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], index=["Prior", "Posterior", "Views"]).T

    # Create EfficientFrontier object
    ef = EfficientFrontier(ret_bl, S)

    # Maximize Sharpe ratio with L2 regularization
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()

    # Calculate optimized portfolio weights
    weights = ef.clean_weights()

    # Display portfolio performance
    performance = ef.portfolio_performance(verbose=True, risk_free_rate=0.009)

    return weights, performance


# In[11]:


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

        subpage = st.sidebar.radio('Select Strategy', ['Low-risk Strategy', 'High-risk Strategy'])

        if subpage == 'Low-risk Strategy':
            st.subheader('Low-risk Portfolio Strategy - Markowitz model')
            tickers = st.multiselect('Select Stock Tickers', all_tickers)
            start_date = st.date_input('Select Start Date:')
            end_date = st.date_input('Select End Date:')
            submit_button = st.button('Submit')
            if submit_button and tickers:
                weights, returns, volatility, sharpe_ratio = build_portfolio(tickers, risk_free_rate, start_date, end_date)

                st.pyplot(plot_portfolio_weights_chart(weights, tickers))
                data = {'Tickers': tickers, 'Weights': weights}
                df = pd.DataFrame(data)
                st.table(df)
                st.write(f'Expected Returns: {round(returns*100,2)}%')
                st.write(f'Expected Volatility: {round(volatility*100,2)}%')
                st.write(f'Sharpe Ratio: {round(sharpe_ratio, 4)}')

        elif subpage == 'High-risk Strategy':
            # BL model
            st.subheader('High-risk Portfolio Strategy - Black-Litterman model')
            tickers = st.multiselect('Select Stock Tickers', all_tickers)
            start_date = st.date_input('Select Start Date:')
            end_date = st.date_input('Select End Date:')
            
            st.subheader('Investor Views')
            investor_views = {}
            for ticker in tickers:
                view = st.number_input(f'Enter investor view for {ticker}:', min_value=-1.0, max_value=1.0, step=0.01)
                investor_views[ticker] = view

            submit_button = st.button('Submit')

            if submit_button and tickers and investor_views:
                weights, performance = black_litterman_portfolio(tickers, investor_views, start_date, end_date)

                # Displayoing the weights as a pie chart
                fig, ax = plt.subplots(figsize=(9, 9))
                pd.Series(weights).plot.pie(ax=ax, autopct='%1.1f%%')
                ax.set_aspect('equal')
                ax.set_title('Portfolio Weights')
                st.pyplot(fig)

                # Displaying the performance metrics
                expected_return, expected_volatility, sharpe_ratio = performance
                st.write("Expected Returns:", round(expected_return*100, 2), "%")
                st.write("Expected Volatility:", round(expected_volatility*100, 2), "%")
                st.write("Sharpe Ratio:", round(sharpe_ratio, 4))

            
        else:
            st.write('Select a strategy from the sidebar')



# In[12]:


if __name__ == "__main__":
    main()


# In[ ]:




