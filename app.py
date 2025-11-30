import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import requests
import xml.etree.ElementTree as ET
import time

# --- Configuration ---
st.set_page_config(
    page_title="ProStock | AI-Powered Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Investing UI ---
st.markdown("""
    <style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: #ffffff; /* Clean white background */
        color: #333333;
    }
    
    /* Logo Header */
    .prostock-logo {
        font-size: 28px;
        font-weight: 900;
        color: #0d6efd;
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .prostock-logo span {
        color: #333;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a1a;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Finance Header Card */
    .finance-header {
        background-color: #ffffff;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 20px;
        margin-bottom: 20px;
    }
    
    /* Key Stats Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .stat-item {
        font-size: 14px;
    }
    .stat-label {
        color: #888;
        font-size: 12px;
    }
    .stat-value {
        font-weight: 600;
        color: #333;
    }
    
    /* Clean News List Styling */
    .news-list-item {
        padding: 12px 0;
        border-bottom: 1px solid #eee;
        display: flex;
        flex-direction: column;
    }
    .news-list-item:hover {
        background-color: #fcfcfc;
    }
    .news-meta {
        font-size: 11px;
        color: #999;
        margin-bottom: 4px;
    }
    .news-link {
        font-size: 15px;
        font-weight: 500;
        color: #1a0dab;
        text-decoration: none;
        line-height: 1.4;
    }
    .news-link:hover {
        text-decoration: underline;
    }
    
    /* Loading Screen */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        animation: fadein 1s;
    }
    .gemini-logo {
        width: 100px;
        margin-top: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    /* Layout Fixes */
    .block-container {
        padding-top: 2rem;
        max-width: 98%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Mock Database ---
@st.cache_resource
def get_database():
    return {}

db = get_database()

# --- Session State ---
if 'user_id' not in st.session_state: st.session_state['user_id'] = None
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'splash_shown' not in st.session_state: st.session_state['splash_shown'] = False
if 'mode' not in st.session_state: st.session_state['mode'] = "Asset Terminal"

# --- Loading Screen ---
if not st.session_state['splash_shown']:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div class="loading-container">
            <h1 style='font-size: 50px; font-weight:900; color: #0d6efd; margin-bottom: 0;'>Pro<span style="color:#333;">Stock</span></h1>
            <p style='color: #666; font-size: 16px;'>Institutional Grade Analytics</p>
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg" class="gemini-logo">
        </div>
        """, unsafe_allow_html=True)
    time.sleep(3)
    placeholder.empty()
    st.session_state['splash_shown'] = True

# --- Auth Functions ---
def login_user(user_id):
    if user_id in db:
        st.session_state['user_id'] = user_id
        st.session_state['logged_in'] = True
        st.success("Access Granted")
        time.sleep(0.5)
        st.rerun()
    else: st.error("ID not found.")

def signup_user(user_id):
    if user_id in db: st.error("ID exists.")
    else:
        db[user_id] = {'favorites': []}
        st.session_state['user_id'] = user_id
        st.session_state['logged_in'] = True
        st.success("Account Created")
        time.sleep(0.5)
        st.rerun()

def logout_user():
    st.session_state['user_id'] = None
    st.session_state['logged_in'] = False
    st.rerun()

def delete_account():
    uid = st.session_state['user_id']
    if uid in db:
        del db[uid]
        st.warning("Account Deleted.")
        logout_user()

# --- LOGIN SCREEN ---
if not st.session_state['logged_in']:
    c1, c2, c3 = st.columns([1,1.5,1])
    with c2:
        st.markdown("""
        <div style='background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; margin-top: 50px;'>
            <h2 style='color: #333;'>ProStock Terminal</h2>
            <p style='color: #777; margin-bottom: 30px;'>Enter 6-Digit ID</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_input("User ID", max_chars=6, type="password", placeholder="######")
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Log In", type="primary", use_container_width=True):
                if len(user_input)==6 and user_input.isdigit(): login_user(user_input)
                else: st.warning("Invalid ID.")
        with b2:
            if st.button("Sign Up", use_container_width=True):
                if len(user_input)==6 and user_input.isdigit(): signup_user(user_input)
                else: st.warning("Invalid ID.")
    st.stop()

# ==========================================
# MAIN APPLICATION
# ==========================================

# --- Helper Functions (Data) ---
@st.cache_data(ttl=60)
def get_stock_data(ticker, interval, period, start=None, end=None):
    try:
        if interval == "1d" and start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if (data.empty or len(data) < 2) and period == "1d":
            data = yf.download(ticker, period="5d", interval=interval, progress=False)
            
        if 'Volume' in data.columns: data = data[data['Volume'] > 0]
        data = data.dropna()
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news
    except: return {}, []

@st.cache_data(ttl=300)
def get_exchange_rate(pair="KRW=X"):
    try:
        data = yf.Ticker(pair).history(period="1d")
        if not data.empty: return data['Close'].iloc[-1]
    except: return None
    return None

def calculate_currency_conversion(amount, from_curr, to_curr):
    if from_curr == to_curr: return amount, 1.0
    try:
        ticker = f"{from_curr}{to_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            return amount * rate, rate
        ticker = f"{to_curr}{from_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = 1.0 / data['Close'].iloc[-1]
            return amount * rate, rate
    except: pass
    return None, None

def calculate_technicals(data):
    if len(data) < 2: return data
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * std
    data['BB_Lower'] = data['BB_Middle'] - 2 * std
    return data

def get_fear_and_greed_proxy():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        sp500 = yf.Ticker("^GSPC").history(period="6mo")
        if sp500.empty: return 50, "Neutral"
        current_sp = sp500['Close'].iloc[-1]
        avg_sp = sp500['Close'].mean()
        fear_score = max(0, min(100, 100 - (vix - 10) * 2.5))
        momentum_score = max(0, min(100, 50 + ((current_sp - avg_sp) / avg_sp) * 500))
        final_score = (fear_score * 0.4) + (momentum_score * 0.6)
        if final_score < 25: label = "Extreme Fear"
        elif final_score < 45: label = "Fear"
        elif final_score < 55: label = "Neutral"
        elif final_score < 75: label = "Greed"
        else: label = "Extreme Greed"
        return int(final_score), label
    except: return 50, "Neutral"

def safe_extract_news_title(item):
    if not isinstance(item, dict): return None
    if 'title' in item and item['title']: return item['title']
    if 'content' in item and isinstance(item['content'], dict):
        if 'title' in item['content'] and item['content']['title']: return item['content']['title']
    for key, value in item.items():
        if isinstance(value, dict):
            res = safe_extract_news_title(value)
            if res: return res
    return None

def analyze_news_sentiment(news_items):
    if not news_items: return 0, 0, 0, "Neutral"
    polarities = []
    for item in news_items:
        title = safe_extract_news_title(item)
        if title:
            blob = TextBlob(title)
            polarities.append(blob.sentiment.polarity)
    if not polarities: return 0, 0, 0, "Neutral"
    pos = sum(1 for p in polarities if p > 0.05)
    neg = sum(1 for p in polarities if p < -0.05)
    neu = len(polarities) - pos - neg
    avg_pol = np.mean(polarities)
    if avg_pol > 0.05: label = "Positive"
    elif avg_pol < -0.05: label = "Negative"
    else: label = "Neutral"
    return pos, neg, neu, label

def generate_ai_report(ticker, price, sma, rsi, fg_score, fg_label, news_label):
    report = f"### 🧠 AI Executive Summary for {ticker}\n\n"
    report += f"**1. Market Sentiment:** {fg_label} ({fg_score}/100).\n"
    report += f"**2. News Analysis:** {news_label} sentiment detected.\n"
    trend = "Bullish 🟢" if price > sma else "Bearish 🔴"
    rsi_state = "Overbought ⚠️" if rsi > 70 else "Oversold 🛒" if rsi < 30 else "Neutral ⚖️"
    report += f"**3. Technicals:** {trend} trend, RSI is {rsi_state}."
    return report

@st.cache_data(ttl=600)
def fetch_rss_feed(url):
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall('.//item')[:10]:
            items.append({
                'title': item.find('title').text,
                'link': item.find('link').text,
                'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else "Recent"
            })
        return items
    except: return []

# --- Sidebar ---
st.sidebar.markdown(f"### 👤 User: {st.session_state['user_id']}")
with st.sidebar.expander("⚙️ Account Center"):
    st.write(f"**ID:** {st.session_state['user_id']}")
    if st.button("Delete Account", type="primary"):
        delete_account()
    if st.button("Log Out"):
        logout_user()

st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 Navigation")

# --- NAVIGATION BUTTONS (Replacing Radio) ---
if st.sidebar.button("📈 Asset Terminal", use_container_width=True):
    st.session_state['mode'] = "Asset Terminal"
if st.sidebar.button("⭐ Favorites", use_container_width=True):
    st.session_state['mode'] = "⭐ Favorites"
if st.sidebar.button("📺 Media & News", use_container_width=True):
    st.session_state['mode'] = "Media & News"

mode = st.session_state['mode']

st.sidebar.markdown("---")

# --- CURRENCY CONVERTER (Moved Up) ---
with st.sidebar.expander("🧮 Currency Calc", expanded=False):
    cc_amt = st.number_input("Amt", 100.0)
    c1, c2 = st.columns(2)
    with c1: cc_f = st.selectbox("From", ["USD", "KRW", "EUR", "JPY", "BTC"])
    with c2: cc_to = st.selectbox("To", ["KRW", "USD", "EUR", "JPY", "BTC"])
    if st.button("Convert"):
        res, rate = calculate_currency_conversion(cc_amt, cc_f, cc_to)
        if res: st.success(f"{res:,.2f} {cc_to}")

# --- FAVORITES ---
if mode == "⭐ Favorites":
    st.title("⭐ My Watchlist")
    user_favs = db[st.session_state['user_id']]['favorites']
    if not user_favs:
        st.info("No favorites yet.")
    else:
        fav_data = []
        for sym in user_favs:
            try:
                d = yf.Ticker(sym).history(period="1d")
                p = d['Close'].iloc[-1] if not d.empty else 0
                fav_data.append({"Ticker": sym, "Price": f"${p:,.2f}"})
            except: fav_data.append({"Ticker": sym, "Price": "Error"})
        st.dataframe(pd.DataFrame(fav_data), use_container_width=True, hide_index=True)

# --- TERMINAL ---
elif mode == "Asset Terminal":
    # Logo
    st.markdown('<div class="prostock-logo">Pro<span>Stock</span></div>', unsafe_allow_html=True)

    # --- Asset Selector ---
    market_type = st.sidebar.selectbox("Market Type", ["Stocks", "Commodities", "Currencies/Forex", "Crypto"])
    ticker = ""
    
    if market_type == "Stocks":
        ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    elif market_type == "Commodities":
        commodities = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Copper": "HG=F", "Natural Gas": "NG=F", "Corn": "ZC=F", "Soybeans": "ZS=F"}
        selected_comm = st.sidebar.selectbox("Select Commodity", list(commodities.keys()))
        ticker = commodities[selected_comm]
    elif market_type == "Currencies/Forex":
        currencies = {"USD/KRW (Won)": "KRW=X", "EUR/USD": "EURUSD=X", "JPY/USD": "JPY=X", "GBP/USD": "GBPUSD=X"}
        selected_curr = st.sidebar.selectbox("Select Pair", list(currencies.keys()))
        ticker = currencies[selected_curr]
    elif market_type == "Crypto":
        coins = {
            "Bitcoin (BTC)": "BTC-USD",
            "Ethereum (ETH)": "ETH-USD",
            "Solana (SOL)": "SOL-USD",
            "XRP (XRP)": "XRP-USD",
            "Dogecoin (DOGE)": "DOGE-USD",
            "Cardano (ADA)": "ADA-USD",
            "Shiba Inu (SHIB)": "SHIB-USD",
            "Binance Coin (BNB)": "BNB-USD",
            "Avalanche (AVAX)": "AVAX-USD",
            "Chainlink (LINK)": "LINK-USD"
        }
        selected_coin = st.sidebar.selectbox("Select Coin", list(coins.keys()))
        ticker = coins[selected_coin]

    user_favs = db[st.session_state['user_id']]['favorites']
    is_fav = ticker in user_favs
    if st.sidebar.checkbox("⭐ Add to Favorites", value=is_fav):
        if not is_fav: db[st.session_state['user_id']]['favorites'].append(ticker)
    else:
        if is_fav: db[st.session_state['user_id']]['favorites'].remove(ticker)

    # Controls
    timeframe = st.sidebar.selectbox("Interval", ["1 Minute", "5 Minute", "1 Hour", "1 Day"])
    if timeframe == "1 Minute": interval, period = "1m", "1d"
    elif timeframe == "5 Minute": interval, period = "5m", "5d"
    elif timeframe == "1 Hour": interval, period = "1h", "1mo"
    else: interval, period = "1d", "1y"

    if interval == "1d":
        start_date = st.sidebar.date_input("Start", value=datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End", value=datetime.now())
    
    st.sidebar.markdown("### Indicators")
    show_sma = st.sidebar.toggle("SMA", True)
    show_bb = st.sidebar.toggle("Bollinger Bands")
    show_rsi = st.sidebar.toggle("RSI")

    if st.sidebar.button("🔄 Refresh Data", type="primary"): st.rerun()

    # --- Main Display ---
    if ticker:
        s_date = start_date if interval == "1d" else None
        e_date = end_date if interval == "1d" else None
        data = get_stock_data(ticker, interval, period, s_date, e_date)
        info, news = get_stock_info(ticker)

        if data is not None and len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data = calculate_technicals(data)
            
            # Latest Values
            latest = data.iloc[-1]
            curr_p = latest['Close']
            open_p = latest['Open']
            high_p = latest['High']
            low_p = latest['Low']
            vol = latest.get('Volume', 0)
            
            prev_p = data['Close'].iloc[-2] if len(data)>1 else curr_p
            delta = curr_p - prev_p
            pct = (delta/prev_p)*100 if prev_p else 0
            
            curr_code = info.get('currency', 'USD')
            ex_rate = get_exchange_rate("KRW=X")
            price_sub = f"(₩{curr_p*ex_rate:,.0f})" if curr_code=='USD' and ex_rate else ""

            # Professional Header
            st.markdown(f"""
            <div class="finance-header">
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <div>
                        <h1 style="margin:0; font-size:32px;">{ticker}</h1>
                        <p style="margin:0; color:#666; font-size:14px;">{info.get('shortName', ticker)} • {market_type}</p>
                    </div>
                    <div style="text-align:right;">
                        <h1 style="margin:0; font-size:36px; color:{'#00C853' if delta>=0 else '#D50000'};">
                            {curr_code} {curr_p:,.2f}
                        </h1>
                        <p style="margin:0; font-weight:bold; color:{'#00C853' if delta>=0 else '#D50000'};">
                            {delta:+.2f} ({pct:+.2f}%) <span style="color:#888; font-weight:normal;">{price_sub}</span>
                        </p>
                    </div>
                </div>
                <div class="stat-grid">
                    <div class="stat-item"><div class="stat-label">OPEN</div><div class="stat-value">{open_p:,.2f}</div></div>
                    <div class="stat-item"><div class="stat-label">HIGH</div><div class="stat-value">{high_p:,.2f}</div></div>
                    <div class="stat-item"><div class="stat-label">LOW</div><div class="stat-value">{low_p:,.2f}</div></div>
                    <div class="stat-item"><div class="stat-label">PREV CLOSE</div><div class="stat-value">{prev_p:,.2f}</div></div>
                    <div class="stat-item"><div class="stat-label">VOLUME</div><div class="stat-value">{vol:,.0f}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            tabs = st.tabs(["Chart", "AI Analysis", "News", "Data"] + (["Fundamentals"] if market_type=="Stocks" else []))

            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price', increasing_line_color='#00C853', decreasing_line_color='#D50000'))
                if show_sma: fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='#FFA000', width=1), name='SMA'))
                if show_bb:
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='#999', width=1, dash='dot'), name='BB Up'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='#999', width=1, dash='dot'), name='BB Lo'))
                
                rangebreaks = [dict(bounds=["sat", "mon"])] if interval in ['1m', '5m', '1h', '1d'] else []
                fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, xaxis=dict(rangebreaks=rangebreaks), margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                fg_score, fg_label = get_fear_and_greed_proxy()
                pos, neg, neu, news_lbl = analyze_news_sentiment(news)
                
                c1, c2 = st.columns([1,2])
                with c1:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=fg_score, title={'text': f"Market Sentiment"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#333"}, 'steps': [{'range': [0, 40], 'color': "#FF5252"}, {'range': [60, 100], 'color': "#00E676"}]}))
                    fig_g.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20)) # Increased height/margin
                    st.plotly_chart(fig_g, use_container_width=True)
                with c2:
                    st.markdown("#### AI Forecast")
                    if len(data) > 30:
                        df_ml = data[['Close']].dropna().reset_index()
                        df_ml['i'] = df_ml.index
                        model = LinearRegression().fit(df_ml[['i']], df_ml['Close'])
                        fut_x = np.arange(df_ml['i'].iloc[-1]+1, df_ml['i'].iloc[-1]+31).reshape(-1,1)
                        pred = model.predict(fut_x)
                        
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=df_ml['i'][-50:], y=df_ml['Close'][-50:], name='History'))
                        fig_p.add_trace(go.Scatter(x=fut_x.flatten(), y=pred, name='Forecast', line=dict(dash='dash', color='red')))
                        fig_p.update_layout(height=250, margin=dict(l=0,r=0,t=20,b=0), template="plotly_white")
                        st.plotly_chart(fig_p, use_container_width=True)
                        st.caption(f"Projected Trend: **{curr_code} {pred[-1]:.2f}**")
                    else: st.warning("Insufficient data for forecast")

                report = generate_ai_report(ticker, curr_p, data['SMA'].iloc[-1], data['RSI'].iloc[-1], fg_score, fg_label, news_lbl)
                st.markdown(f"""<div style="background:#f8f9fa; padding:20px; border-radius:5px; border-left:4px solid #0d6efd;">{report.replace(chr(10), '<br>')}</div>""", unsafe_allow_html=True)

            with tabs[2]:
                if news:
                    for i in news[:10]:
                        t = safe_extract_news_title(i) or "News"
                        l = i.get('link') or i.get('url') or "#"
                        pub = i.get('publisher', 'Source')
                        ts = i.get('providerPublishTime', 0)
                        time_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else "Recent"
                        st.markdown(f"""
                        <div class="news-list-item">
                            <div class="news-meta">{pub} • {time_str}</div>
                            <a href="{l}" target="_blank" class="news-link">{t}</a>
                        </div>
                        """, unsafe_allow_html=True)
                else: st.info("No news.")

            with tabs[3]: st.dataframe(data.tail(50), use_container_width=True)
            
            if market_type == "Stocks":
                with tabs[4]:
                    st.subheader("Company Profile")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Country:** {info.get('country', 'N/A')}")
                        st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                    with c2:
                        st.write(f"**Website:** {info.get('website', 'N/A')}")
                        st.write(f"**City:** {info.get('city', 'N/A')}")
                        st.write(f"**Phone:** {info.get('phone', 'N/A')}")
                    st.markdown("---")
                    st.write("**Business Summary:**")
                    st.write(info.get('longBusinessSummary', 'N/A'))

# --- MEDIA MODE ---
elif mode == "Media & News":
    st.title("📺 Global Media Center")
    st.subheader("Quick Access")
    
    # Text-Based Buttons for Quick Access (More Reliable than Images)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        st.link_button("🌐 Investing.com", "https://www.investing.com", use_container_width=True)
    with qa2:
        st.link_button("📈 Yahoo Finance", "https://finance.yahoo.com", use_container_width=True)
    with qa3:
        st.link_button("🔎 Google Finance", "https://www.google.com/finance", use_container_width=True)

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bloomberg TV")
        st.video("https://www.youtube.com/watch?v=iEpJwprxDdk")
        st.subheader("Sky News")
        st.video("https://www.youtube.com/watch?v=YDvsBbKfLPA")
    with c2:
        st.subheader("CNA Asia")
        st.video("https://www.youtube.com/watch?v=XWq5kBlakcQ")
        st.subheader("ABC Australia")
        st.video("https://www.youtube.com/watch?v=iipR5yUp36o")
    
    st.markdown("---")
    st.subheader("Live Wires")
    
    def get_feed(url):
        try:
            r = requests.get(url, timeout=3)
            root = ET.fromstring(r.content)
            return [{'t':i.find('title').text, 'l':i.find('link').text} for i in root.findall('.//item')[:5]]
        except: return []

    t1, t2, t3 = st.tabs(["CNBC", "BBC", "CNN"])
    with t1:
        for n in get_feed("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"):
            st.markdown(f"<div class='news-list-item'><a href='{n['l']}' target='_blank' class='news-link'>{n['t']}</a></div>", unsafe_allow_html=True)
    with t2:
        for n in get_feed("http://feeds.bbci.co.uk/news/business/rss.xml"):
            st.markdown(f"<div class='news-list-item'><a href='{n['l']}' target='_blank' class='news-link'>{n['t']}</a></div>", unsafe_allow_html=True)
    with t3:
        for n in get_feed("http://rss.cnn.com/rss/money_latest.rss"):
            st.markdown(f"<div class='news-list-item'><a href='{n['l']}' target='_blank' class='news-link'>{n['t']}</a></div>", unsafe_allow_html=True)
