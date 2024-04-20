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

# In[29]:


def get_stock_data(tickers, start_date, end_date):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers_str = ' '.join(tickers)
    data = yf.download(tickers_str, start=start_date, end=end_date)
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


# In[19]:


def forecast_stock_price(data, ticker, investor_view=False, start_date=None, end_date=None):

    if investor_view:
            if start_date and end_date:
                data = data.reset_index()
                df = data[['Date', 'Adj Close']]
                df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)
                model = Prophet()
                model.fit(df)            
                forecast_date = end_date + timedelta(days=(end_date - start_date).days // 4)
                future = pd.DataFrame({'ds': [forecast_date]})
                inv_view = model.predict(future)
                return inv_view
            else:
                st.warning("Please provide start date and end date to calculate investor view.")
                return None
                
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
        st.warning("Not enough historical data available for forecasting. (Select at least **twelve months** of data to forecast)")


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
    ax.set_title('Optimal Portfolio Weights')
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

# In[48]:


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
    performance = ef.portfolio_performance(verbose=True, risk_free_rate = get_risk_free_rate())

    return weights, performance


# In[61]:


def homepage():
    
    st.markdown("""
        <style>
            html, body, .stApp, .stApp > div, .stApp > div > div {
                width: 100%;
                margin: 0;
                padding: 0;
            }
            
            .content {
                color: white;
                text-align: left; 
                padding-left: 0 !important; 
                margin-left: 0 !important; 
            }
            
            .header {
                display: flex;
                align-items: center;
            }
    
            h1 {
                font-size: 2.75em;
                font-weight: bold;
            }
            
            p {
                font-size: 1.3em;
                margin-top: 30px;
            }

            .favicon {
                width: 35px;
                height: auto;
                vertical-align: middle;
            }
            
        </style>
        <div class="content">
            <div class="header">
                <img src="https://www.favicon.cc/logo3d/737096.png" class="favicon" alt="SmartFolio Logo">
                <h1 style="color: white;">SmartFolio - Portfolio Optimizer</h1>
            </div>
            <p>A simple application designed to assist you in forecasting and visualizing stock performances, and to help build your portfolio using optimization techniques.</p>
            <p>Begin by selecting an option from the left navigation pane.</p>
        </div>
        """, unsafe_allow_html=True)


# In[59]:


def explore_stocks_page(all_tickers):
    max_end_date = datetime.today()
    
    st.subheader('View Stock Data and Make Forecasts')  
    selected_stocks = st.multiselect('Select Stock Tickers', all_tickers)        
    start_date = st.date_input('Select Start Date:', max_value=max_end_date)
    end_date = st.date_input('Select End Date:', max_value=max_end_date)

    view_charts = st.button('View Charts')
    view_forecasts = st.button('Forecast')

    if view_charts:
        for stock in selected_stocks:
            ticker = stock.split('-')[1].strip()
            data = get_stock_data(ticker, start_date, end_date)
            if not data.empty:
                plot_chart(data, ticker)
            else:
                st.warning(f'No data found for {ticker} for the selected timeframe.')

    if view_forecasts:
        for stock in selected_stocks:
            ticker = stock.split('-')[1].strip()
            data = get_stock_data(ticker, start_date, end_date)
            if not data.empty:
                st.subheader(f'Forecast for {ticker}')
                forecast_stock_price(data, ticker)
            else:
                st.warning(f'No data found for {ticker} for the selected timeframe.')


# In[60]:


def build_portfolio_page(all_tickers, risk_free_rate):
    subpage = st.sidebar.radio('Select Strategy', ['Low-risk Strategy', 'High-risk Strategy'])

    max_end_date = datetime.today()

    if subpage == 'Low-risk Strategy':
        st.subheader('Low-risk Portfolio Strategy - Markowitz model')
        selected_stocks = st.multiselect('Select Stock Tickers', all_tickers)
        tickers = [stock.split('-')[1].strip() for stock in selected_stocks]
        start_date = st.date_input('Select Start Date:', max_value=max_end_date)
        end_date = st.date_input('Select End Date:', max_value=max_end_date)
        init_capital = st.number_input("Enter your initial capital (USD):", min_value=0.0, format="%.2f")

        if start_date == end_date or (end_date-start_date).days<90:
            st.write('Please select a start date at least **3 months** in the past for accurate results!')
        else:
            submit_button = st.button('Submit')
            if submit_button and tickers and init_capital>0:
                weights, returns, volatility, sharpe_ratio = build_portfolio(tickers, risk_free_rate, start_date, end_date)
                expected_change = init_capital + init_capital * returns
                st.pyplot(plot_portfolio_weights_chart(weights, tickers))
                st.markdown(f"**Expected Returns:** {round(returns * 100, 2)}%")
                st.markdown(f"**Expected Volatility:** {round(volatility * 100, 2)}%")
                st.markdown(f"**Sharpe Ratio:** {round(sharpe_ratio, 2)}")
                st.markdown(f"**Expected portfolio balance:** ${expected_change:.2f}")
            else:
                st.write('Before you hit Submit, Please ensure you have selected at least 1 stock and entered your initial capital!')

    elif subpage == 'High-risk Strategy':
        # BL model
        st.subheader('High-risk Portfolio Strategy - Black-Litterman model')
        selected_stocks = st.multiselect('Select Stock Tickers', all_tickers)
        tickers = [stock.split('-')[1].strip() for stock in selected_stocks]
        start_date = st.date_input('Select Start Date:', max_value=max_end_date)
        end_date = st.date_input('Select End Date:', max_value=max_end_date)
        init_capital = st.number_input("Enter your initial capital (USD):", min_value=0.0, format="%.2f")

        if start_date == end_date or (end_date-start_date).days<90:
            st.write('Please select a start date at least **3 months** in the past for accurate results!')
        else:
            data_BL_all = {ticker: get_stock_data(ticker, start_date, end_date) for ticker in tickers}
            
            st.subheader('Investor Views')
            investor_views = {}
            
            # Calculate default investor view
            for ticker in tickers:
                data_BL = data_BL_all[ticker]
                forecasted_ticker_price = forecast_stock_price(data_BL, ticker, investor_view=True, start_date=start_date, end_date=end_date)['yhat']
                end_date_ticker_price = data_BL['Adj Close'].iloc[-1]
                net_change = ((forecasted_ticker_price - end_date_ticker_price) / end_date_ticker_price)
                view = st.number_input(f'Enter your view for {ticker}: (My views: {round(net_change.item(),2)})', value=net_change.item(), min_value=-1.0, max_value=1.0, step=0.01)
                investor_views[ticker] = view

            submit_button = st.button('Submit')

            if submit_button and tickers and investor_views and init_capital>0:
                weights, performance = black_litterman_portfolio(tickers, investor_views, start_date, end_date)

                # Displayoing the weights as a pie chart
                fig, ax = plt.subplots(figsize=(9, 9))
                pd.Series(weights).plot.pie(ax=ax, autopct='%1.1f%%')
                ax.set_aspect('equal')
                ax.set_title('Portfolio Weights')
                st.pyplot(fig)

                # Displaying the performance metrics
                expected_return, expected_volatility, sharpe_ratio = performance
                expected_change = init_capital + init_capital * expected_return
                st.markdown(f"**Expected Returns:** {round(expected_return * 100, 2)}%")
                st.markdown(f"**Expected Volatility:** {round(expected_volatility * 100, 2)}%")
                st.markdown(f"**Sharpe Ratio:** {round(sharpe_ratio, 2)}")
                st.markdown(f"**Expected portfolio balance:** ${expected_change:.2f}")

            else:
                st.write('Before you hit Submit, Please ensure you have selected at least 1 stock and entered your initial capital!')
    
    else:
        st.write('Select a strategy from the sidebar')

    pass


# In[67]:


def set_background():
    st.markdown("""
        <style>
            .stApp {
                background-image: url("https://www.colorhexa.com/2f2f4f.png");
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        </style>
        """, unsafe_allow_html=True)

def apply_white_text_style():
    st.markdown("""
        <style>
            /* Targeting the main body directly, might override some Streamlit-specific styles */
            body {
                color: white;
            }
            .stTextInput > label, .stDateInput > label, .stNumberInput > label {
                color: white;
            }
            .stMarkdown {
                color: white; 
            }
            h3 {
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)


# In[63]:


def main():
    st.set_page_config(layout="wide")
    set_background()
    
    risk_free_rate = get_risk_free_rate()
    company_ticker_map = {}
    with open('all_tickers.txt', 'r') as file:
        for line in file:
            company_name, ticker = line.split('\t')
            company_ticker_map[company_name.strip()] = ticker.strip()

    all_tickers = [f"{company_name} - {ticker}" for company_name, ticker in company_ticker_map.items()]

    st.sidebar.title('Navigate')

    page = st.sidebar.selectbox('Page', ["Home", "Explore Stocks", "Build Your Portfolio"])

    if page == "Home":

        homepage()

    if page == "Explore Stocks":
        
        apply_white_text_style()
        explore_stocks_page(all_tickers)

    if page == "Build Your Portfolio":

        apply_white_text_style()
        build_portfolio_page(all_tickers, risk_free_rate)

    # Author info and links section
    st.sidebar.markdown("""<br><br><br><br><br><br><br><br><br><br><br><br>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <style>

        .info-box {
            border: 1px solid #3B7080; 
            color: #20435C; 
            padding: 10px;
            background-color: #E6EFF1;  
            border-radius: 5px;
        }
        .info-box a {
            color: #20435C; 
            text-decoration: none;
        }
        .info-box a:hover {
            color: #3B7080;
            text-decoration: underline;
        }
    </style>
    <div class="info-box">
        <h4>About Me:</h4>
        <a href="https://github.com/ark097" target="_blank">My Github</a><br>
        <a href="https://github.com/ark097/Portfolio_Opt" target="_blank">Code</a>
    </div>
    """, unsafe_allow_html=True)


# In[40]:


if __name__ == "__main__":
    main()

