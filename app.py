import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import webbrowser
import requests
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="INVESTO - Smart Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4037;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header"> I.N.V.E.S.T.O</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Intelligent Investment Companion</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header(" Control Panel")
    
    # Stock selection
    st.subheader(" Stock Selection")
    
    # Load S&P 500 ticker list
    @st.cache_data
    def load_sp500_tickers():
        try:
            url = 'https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt'
            response = requests.get(url)
            response.raise_for_status()  # Raise error if request failed
            ticker_list = response.text.strip().split('\n')  # Split by lines
            return ticker_list
        except Exception as e:
            print("Error fetching ticker list:", e)
            # Fallback popular stocks if URL fails
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "NFLX"]
    
    sp500_tickers = load_sp500_tickers()
    
    # Popular stocks for quick access (subset of S&P 500)
    popular_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "JNJ"]
    
    stock_choice = st.radio("Choose stock selection method:", ["S&P 500 Stocks", "Popular Picks", "Manual Entry"])
    
    if stock_choice == "S&P 500 Stocks":
        tickerSymbol = st.selectbox(
            "Select from S&P 500 companies:", 
            sp500_tickers,
            index=sp500_tickers.index("AAPL") if "AAPL" in sp500_tickers else 0
        )
        st.success(f" Selected: **{tickerSymbol}** (S&P 500 Company)")
        
    elif stock_choice == "Popular Picks":
        tickerSymbol = st.selectbox("Select a popular stock:", popular_stocks)
        st.info(f" Selected: **{tickerSymbol}** (Popular Pick)")
        
    else:
        tickerSymbol = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", value="AAPL").upper()
        if tickerSymbol in sp500_tickers:
            st.success(f" **{tickerSymbol}** is an S&P 500 company")
        else:
            st.warning(f" **{tickerSymbol}** is not in S&P 500 (but may still be valid)")
    
    # Date range selection
    st.subheader(" Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    # Quick date range buttons
    st.subheader(" Quick Selection")
    if st.button(" Last 30 Days", use_container_width=True):
        start_date = datetime.date.today() - datetime.timedelta(days=30)
    if st.button(" Last 6 Months", use_container_width=True):
        start_date = datetime.date.today() - datetime.timedelta(days=180)
    if st.button(" Last Year", use_container_width=True):
        start_date = datetime.date.today() - datetime.timedelta(days=365)

# Function to load stock data
@st.cache_data
def load_stock_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        info = stock.info
        return data, info, None
    except Exception as e:
        return None, None, str(e)

# Function to calculate technical indicators
def calculate_indicators(data):
    # Moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    data['BB_upper'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_lower'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Load data
if tickerSymbol:
    with st.spinner(f"Loading data for {tickerSymbol}..."):
        stock_data, stock_info, error = load_stock_data(tickerSymbol, start_date, end_date)
    
    if error:
        st.error(f"Error loading data: {error}")
        st.stop()
    
    if stock_data is not None and not stock_data.empty:
        # Calculate technical indicators
        stock_data = calculate_indicators(stock_data)
        
        # Display basic stock info
        if stock_info:
            st.subheader(f" {stock_info.get('longName', tickerSymbol)} ({tickerSymbol})")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(" Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            with col2:
                st.metric(" Market Cap", f"${stock_info.get('marketCap', 0):,.0f}")
            with col3:
                st.metric("P/E Ratio", f"{stock_info.get('trailingPE', 'N/A')}")
            with col4:
                st.metric(" 52W High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
            with col5:
                st.metric(" 52W Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([" Price Analysis", " Technical Analysis", " Predictions", " Financial Metrics", " Trading Platforms"])
        
        with tab1:
            st.subheader(" Price Movement Analysis")
            
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Chart', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Moving averages
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['MA20'], name='MA20', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['MA50'], name='MA50', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            fig.update_layout(height=700, title=f"{tickerSymbol} Stock Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent data table
            st.subheader(" Recent Data")
            st.dataframe(stock_data.tail(10).round(2), use_container_width=True)
        
        with tab2:
            st.subheader(" Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bollinger Bands
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
                fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_upper'], name='Upper Band', line=dict(dash='dash')))
                fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_lower'], name='Lower Band', line=dict(dash='dash')))
                fig_bb.update_layout(title="Bollinger Bands", height=400)
                st.plotly_chart(fig_bb, use_container_width=True)
            
            with col2:
                # RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI (Relative Strength Index)", height=400)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Technical Analysis Summary
            st.subheader(" Technical Indicators Summary")
            current_rsi = stock_data['RSI'].iloc[-1]
            current_price = stock_data['Close'].iloc[-1]
            ma20 = stock_data['MA20'].iloc[-1]
            ma50 = stock_data['MA50'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                st.metric("RSI Signal", f"{current_rsi:.1f}", rsi_signal)
            
            with col2:
                ma_signal = "🟢 Bullish" if current_price > ma20 > ma50 else "🔴 Bearish"
                st.metric("MA Trend", ma_signal)
            
            with col3:
                volatility = stock_data['Close'].pct_change().std() * 100
                st.metric("Volatility", f"{volatility:.2f}%")
        
        with tab3:
            st.subheader(" Price Predictions")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                n_years = st.slider('Prediction Period (Years):', 0.5, 3.0, 1.0, 0.5)
                prediction_days = int(n_years * 365)
                
                if st.button(" Generate Prediction", use_container_width=True):
                    with st.spinner("Training AI model..."):
                        # Prepare data for Prophet
                        df_train = stock_data.reset_index()[['Date', 'Close']]
                        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                        
                        # Remove timezone information from the date column
                        df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)
                        
                        # Train model
                        model = Prophet(daily_seasonality=True)
                        model.fit(df_train)
                        
                        # Make predictions
                        future = model.make_future_dataframe(periods=prediction_days)
                        forecast = model.predict(future)
                        
                        # Store in session state
                        st.session_state.forecast = forecast
                        st.session_state.model = model
                        st.session_state.prediction_generated = True
            
            with col2:
                if hasattr(st.session_state, 'prediction_generated') and st.session_state.prediction_generated:
                    # Display prediction chart
                    fig_forecast = plot_plotly(st.session_state.model, st.session_state.forecast)
                    fig_forecast.update_layout(title=f"{tickerSymbol} Price Prediction", height=500)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show prediction summary
                    last_actual = st.session_state.forecast[st.session_state.forecast['ds'] <= pd.Timestamp.today()]['yhat'].iloc[-1]
                    future_price = st.session_state.forecast['yhat'].iloc[-1]
                    price_change = ((future_price - last_actual) / last_actual) * 100
                    
                    st.info(f" Predicted price in {n_years} year(s): ${future_price:.2f} ({price_change:+.1f}%)")
                else:
                    st.info(" Click 'Generate Prediction' to see AI-powered price forecasts")
        
        with tab4:
            st.subheader(" Financial Metrics & Analysis")
            
            if stock_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("** Valuation Metrics**")
                    metrics_data = {
                        "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
                        "Forward P/E": stock_info.get('forwardPE', 'N/A'),
                        "PEG Ratio": stock_info.get('pegRatio', 'N/A'),
                        "Price to Book": stock_info.get('priceToBook', 'N/A'),
                        "Price to Sales": stock_info.get('priceToSalesTrailing12Months', 'N/A')
                    }
                    
                    for metric, value in metrics_data.items():
                        if value != 'N/A' and value is not None:
                            st.write(f"• **{metric}:** {value:.2f}")
                        else:
                            st.write(f"• **{metric}:** N/A")
                
                with col2:
                    st.write("** Financial Health**")
                    financial_data = {
                        "Revenue": stock_info.get('totalRevenue', 'N/A'),
                        "Profit Margin": stock_info.get('profitMargins', 'N/A'),
                        "Operating Margin": stock_info.get('operatingMargins', 'N/A'),
                        "Return on Equity": stock_info.get('returnOnEquity', 'N/A'),
                        "Debt to Equity": stock_info.get('debtToEquity', 'N/A')
                    }
                    
                    for metric, value in financial_data.items():
                        if metric == "Revenue" and value != 'N/A' and value is not None:
                            st.write(f"• **{metric}:** ${value:,.0f}")
                        elif value != 'N/A' and value is not None:
                            if isinstance(value, float):
                                st.write(f"• **{metric}:** {value:.2%}" if metric in ["Profit Margin", "Operating Margin", "Return on Equity"] else f"• **{metric}:** {value:.2f}")
                            else:
                                st.write(f"• **{metric}:** {value}")
                        else:
                            st.write(f"• **{metric}:** N/A")
                
                # Company description
                if 'longBusinessSummary' in stock_info:
                    st.subheader(" Company Overview")
                    st.write(stock_info['longBusinessSummary'])
        
        with tab5:
            st.subheader(" Trading Platforms")
            st.write("Ready to invest? Choose your preferred trading platform:")
            
            platforms = {
                "Zerodha": {
                    "url": "https://zerodha.com/",
                    "description": "India's largest stock broker with zero brokerage on equity delivery trades",
                    "features": ["Zero brokerage on delivery", "Advanced trading tools", "Educational resources"]
                },
                "Groww": {
                    "url": "https://groww.in/",
                    "description": "Simple and intuitive platform for beginners",
                    "features": ["User-friendly interface", "Mutual funds", "No account opening charges"]
                },
                "5paisa": {
                    "url": "https://www.5paisa.com/",
                    "description": "Low-cost trading with comprehensive research tools",
                    "features": ["Low brokerage", "Research reports", "Mobile trading app"]
                },
                "ET Money": {
                    "url": "https://www.etmoney.com/",
                    "description": "Complete financial services platform",
                    "features": ["Tax planning", "Insurance", "Investment tracking"]
                }
            }
            
            for platform, details in platforms.items():
                with st.expander(f"🔗 {platform}"):
                    st.write(details["description"])
                    st.write("**Key Features:**")
                    for feature in details["features"]:
                        st.write(f"• {feature}")
                    
                    if st.button(f"Visit {platform}", key=platform, use_container_width=True):
                        webbrowser.open_new_tab(details["url"])
                        st.success(f"Opening {platform} in a new tab...")
    
    else:
        st.error(f"No data found for ticker '{tickerSymbol}'. Please check the ticker symbol and try again.")

else:
    st.info(" Please enter a stock ticker symbol in the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p> INVESTO - Your Intelligent Investment Companion | Built with  using Streamlit</p>
        <p><em>Disclaimer: This tool is for educational purposes only. Always consult with a financial advisor before making investment decisions.</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)