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

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #ffffff; color: #333333; }
    
    /* Logo Styles */
    .prostock-logo-sidebar { font-size: 24px; font-weight: 900; color: #0d6efd; margin-bottom: 20px; cursor: pointer; }
    .prostock-logo-sidebar span { color: #333; }
    
    .homepage-logo { font-size: 60px; font-weight: 900; color: #0d6efd; text-align: center; margin-bottom: 10px; }
    .homepage-logo span { color: #333; }
    
    /* Homepage Elements */
    .hero-container { padding: 40px 20px; text-align: center; }
    .big-search-container { max-width: 700px; margin: 0 auto 40px auto; }
    
    /* Trending Cards */
    .trend-card {
        background: white; border: 1px solid #eee; border-radius: 8px; padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: 0.2s;
    }
    .trend-card:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .trend-header { font-size: 14px; color: #666; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; }
    .trend-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; font-size: 14px; }
    .trend-item:last-child { border-bottom: none; }
    .trend-name { font-weight: 500; color: #0d6efd; }
    .trend-price { font-weight: 600; }
    
    /* Account Top Right */
    .account-bar {
        display: flex; justify-content: flex-end; align-items: center; gap: 15px;
        padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 20px;
    }
    .user-badge { font-weight: 600; color: #555; background: #e9ecef; padding: 5px 10px; border-radius: 20px; font-size: 12px; }
    
    /* Existing Styles */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    .finance-header { background-color: #ffffff; border-bottom: 2px solid #f0f0f0; padding-bottom: 20px; margin-bottom: 20px; margin-top: 10px; }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 15px; margin-top: 15px; }
    .stat-item { font-size: 14px; }
    .stat-label { color: #888; font-size: 12px; }
    .stat-value { font-weight: 600; color: #333; }
    .news-list-item { padding: 12px 0; border-bottom: 1px solid #eee; display: flex; flex-direction: column; }
    .news-link { font-size: 15px; font-weight: 500; color: #1a0dab; text-decoration: none; }
    
    /* Loading */
    .loading-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh; animation: fadein 1s; }
    .gemini-logo { width: 100px; margin-top: 20px; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.05); opacity: 1; } 100% { transform: scale(1); opacity: 0.8; } }
    
    .block-container { padding-top: 2rem; max-width: 98%; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Mock DB & State ---
@st.cache_resource
def get_database(): return {}
db = get_database()

if 'user_id' not in st.session_state: st.session_state['user_id'] = None
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'splash_shown' not in st.session_state: st.session_state['splash_shown'] = False
if 'mode' not in st.session_state: st.session_state['mode'] = "Home" # Default to Home
if 'ticker_search' not in st.session_state: st.session_state['ticker_search'] = ""

# --- Loading Screen ---
if not st.session_state['splash_shown']:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""<div class="loading-container"><h1 style='font-size: 50px; font-weight:900; color: #0d6efd;'>Pro<span style="color:#333;">Stock</span></h1><p style='color: #666;'>Institutional Grade Analytics</p><img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg" class="gemini-logo"></div>""", unsafe_allow_html=True)
    time.sleep(3)
    placeholder.empty()
    st.session_state['splash_shown'] = True

# --- Auth ---
def login_user(uid):
    if uid in db:
        st.session_state['user_id'] = uid; st.session_state['logged_in'] = True; st.rerun()
    else: st.error("ID not found")
def signup_user(uid):
    if uid in db: st.error("ID exists")
    else: db[uid]={'favorites':[]}; st.session_state['user_id']=uid; st.session_state['logged_in']=True; st.rerun()
def logout_user():
    st.session_state['user_id']=None; st.session_state['logged_in']=False; st.rerun()
def delete_account():
    if st.session_state['user_id'] in db: del db[st.session_state['user_id']]; logout_user()

if not st.session_state['logged_in']:
    c1,c2,c3 = st.columns([1,1.5,1])
    with c2:
        st.markdown("""<div style='background:white;padding:40px;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.08);text-align:center;margin-top:50px;'><h2 style='color:#333;'>ProStock Terminal</h2><p style='color:#777;margin-bottom:30px;'>Enter 6-Digit ID</p></div>""", unsafe_allow_html=True)
        uid = st.text_input("User ID", max_chars=6, type="password", placeholder="######")
        b1,b2=st.columns(2)
        with b1: 
            if st.button("Log In", type="primary", use_container_width=True): 
                if len(uid)==6 and uid.isdigit(): login_user(uid)
        with b2:
            if st.button("Sign Up", use_container_width=True):
                if len(uid)==6 and uid.isdigit(): signup_user(uid)
    st.stop()

# --- Common Data Functions ---
ASSET_MAP = {
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
    "SOLANA": "SOL-USD", "XRP": "XRP-USD", "GOLD": "GC=F", "SILVER": "SI=F",
    "OIL": "CL=F", "USD/KRW": "KRW=X", "APPLE": "AAPL", "TESLA": "TSLA",
    "NVIDIA": "NVDA", "GOOGLE": "GOOGL", "AMAZON": "AMZN", "SAMSUNG": "005930.KS"
}

@st.cache_data(ttl=60)
def get_live_price(ticker):
    try:
        d = yf.Ticker(ticker).history(period="1d")
        if not d.empty:
            price = d['Close'].iloc[-1]
            prev = d['Open'].iloc[0] # Approx for intraday change
            change = ((price - prev)/prev)*100
            return price, change
    except: pass
    return 0.0, 0.0

# --- NAVIGATION ---
# Sidebar Logo acts as "Home" button
if st.sidebar.button("📈 ProStock", type="primary", use_container_width=True):
    st.session_state['mode'] = "Home"

st.sidebar.markdown("---")
if st.sidebar.button("📈 Asset Terminal", use_container_width=True): st.session_state['mode'] = "Asset Terminal"
if st.sidebar.button("⭐ Favorites", use_container_width=True): st.session_state['mode'] = "Favorites"
if st.sidebar.button("📺 Media & News", use_container_width=True): st.session_state['mode'] = "Media & News"

mode = st.session_state['mode']

st.sidebar.markdown("---")

# --- CURRENCY CONVERTER ---
with st.sidebar.expander("🧮 Currency Calc", expanded=False):
    cc_amt = st.number_input("Amt", 100.0)
    c1, c2 = st.columns(2)
    with c1: cc_f = st.selectbox("From", ["USD", "KRW", "EUR", "JPY", "BTC"])
    with c2: cc_to = st.selectbox("To", ["KRW", "USD", "EUR", "JPY", "BTC"])
    # Helper for simple conversion
    def simple_convert(a, f, t):
        if f==t: return a
        try:
            if f=='USD': r = yf.Ticker(f"{t}=X").history(period='1d')['Close'].iloc[-1]; return a/r if t!='KRW' else a*r # Logic varies by pair convention, simple hack:
            # Fallback robust
            pair = f"{f}{t}=X"
            d = yf.Ticker(pair).history(period='1d')
            if not d.empty: return a * d['Close'].iloc[-1]
            # Inverse
            pair = f"{t}{f}=X"
            d = yf.Ticker(pair).history(period='1d')
            if not d.empty: return a * (1/d['Close'].iloc[-1])
        except: return None
        return None

    if st.button("Convert"):
        res = simple_convert(cc_amt, cc_f, cc_to)
        if res: st.success(f"{res:,.2f} {cc_to}")
        else: st.error("Rate unavailable")

# --- MODE: HOMEPAGE ---
if mode == "Home":
    # Top Right Account Center (Only on Home) - Single Button Style
    c_fill, c_acc = st.columns([3, 1])
    with c_acc:
        with st.expander(f"👤 ID: {st.session_state['user_id']}"):
            if st.button("Log Out", use_container_width=True): logout_user()
            if st.button("Delete Account", type="primary", use_container_width=True): delete_account()

    # Hero Section
    st.markdown("""<div class="hero-container"><div class="homepage-logo">Pro<span>Stock</span></div><p style="font-size:18px; color:#666;">Market Intelligence for the Modern Investor</p></div>""", unsafe_allow_html=True)
    
    # Big Search Bar
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        big_search = st.text_input("🔍 Search markets (e.g. Apple, Bitcoin, Gold...)", placeholder="Search for stocks, crypto, or currencies...", label_visibility="collapsed")
        if big_search:
            # Redirect logic
            q_upper = big_search.upper().strip()
            ticker_res = ASSET_MAP.get(q_upper, q_upper)
            st.session_state['ticker_search'] = ticker_res
            st.session_state['mode'] = "Asset Terminal"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Trending Dashboards
    t1, t2, t3 = st.columns(3)
    
    def render_trend_card(title, assets):
        st.markdown(f"""<div class="trend-card"><div class="trend-header">{title}</div>""", unsafe_allow_html=True)
        for name, sym in assets.items():
            p, chg = get_live_price(sym)
            color = "#00C853" if chg >= 0 else "#D50000"
            st.markdown(f"""
            <div class="trend-item">
                <span class="trend-name">{name}</span>
                <span class="trend-price" style="color:{color}">{p:,.2f} ({chg:+.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t1:
        render_trend_card("🔥 Trending Stocks", {"NVIDIA": "NVDA", "Tesla": "TSLA", "Apple": "AAPL", "Amazon": "AMZN"})
    with t2:
        render_trend_card("🪙 Top Crypto", {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD"})
    with t3:
        render_trend_card("💱 Key Currencies", {"USD/KRW": "KRW=X", "EUR/USD": "EURUSD=X", "JPY/USD": "JPY=X", "GBP/USD": "GBPUSD=X"})

# --- MODE: ASSET TERMINAL ---
elif mode == "Asset Terminal":
    # Initialize ticker from session state if coming from search
    default_ticker = st.session_state.get('ticker_search', "")
    
    # Layout: Sidebar controls still active
    st.markdown('<div class="prostock-logo" style="font-size:24px;">Pro<span>Stock</span> Terminal</div>', unsafe_allow_html=True)
    
    # Top Bar Search (Smaller)
    search_query = st.text_input("Search Assets", value=default_ticker, placeholder="Symbol or Name...", label_visibility="collapsed")
    
    # Resolve Ticker
    ticker = ""
    market_type = "Stocks"
    
    if search_query:
        q_upper = search_query.upper().strip()
        ticker = ASSET_MAP.get(q_upper, q_upper)
        # Auto-classify
        if ticker.endswith("-USD"): market_type = "Crypto"
        elif ticker.endswith("=F"): market_type = "Commodities"
        elif ticker.endswith("=X"): market_type = "Currencies/Forex"
    else:
        # Fallback to manual selection if search empty
        market_type = st.sidebar.selectbox("Market Type", ["Stocks", "Commodities", "Currencies/Forex", "Crypto"])
        if market_type == "Stocks": ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
        elif market_type == "Commodities":
            commodities = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Copper": "HG=F", "Natural Gas": "NG=F", "Corn": "ZC=F", "Soybeans": "ZS=F"}
            ticker = commodities[st.sidebar.selectbox("Select", list(commodities.keys()))]
        elif market_type == "Currencies/Forex":
            currencies = {"USD/KRW (Won)": "KRW=X", "EUR/USD": "EURUSD=X", "JPY/USD": "JPY=X", "GBP/USD": "GBPUSD=X"}
            ticker = currencies[st.sidebar.selectbox("Select", list(currencies.keys()))]
        elif market_type == "Crypto":
            coins = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD", "XRP": "XRP-USD", "Dogecoin": "DOGE-USD", "Cardano": "ADA-USD"}
            ticker = coins[st.sidebar.selectbox("Select", list(coins.keys()))]

    # Save current ticker to state
    st.session_state['ticker_search'] = ticker

    user_favs = db[st.session_state['user_id']]['favorites']
    is_fav = ticker in user_favs
    if st.sidebar.checkbox("⭐ Add to Favorites", value=is_fav):
        if not is_fav: db[st.session_state['user_id']]['favorites'].append(ticker)
    else:
        if is_fav: db[st.session_state['user_id']]['favorites'].remove(ticker)

    # Controls (Timeframe/Indicators)
    with st.sidebar.expander("⚙️ Chart Settings", expanded=True):
        timeframe = st.selectbox("Interval", ["1 Minute", "5 Minute", "1 Hour", "1 Day"])
        show_sma = st.toggle("SMA", True)
        show_bb = st.toggle("Bollinger Bands")
        show_rsi = st.toggle("RSI")
        
    if timeframe == "1 Minute": interval, period = "1m", "1d"
    elif timeframe == "5 Minute": interval, period = "5m", "5d"
    elif timeframe == "1 Hour": interval, period = "1h", "1mo"
    else: interval, period = "1d", "1y"

    if interval == "1d":
        start_date = st.sidebar.date_input("Start", value=datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End", value=datetime.now())

    if st.sidebar.button("🔄 Refresh Data", type="primary"): st.rerun()

    # Data Fetching & Display
    if ticker:
        try:
            # Fetch Data
            # Handle start/end dates only for daily
            if interval == "1d":
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            else:
                data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            # Fallback for weekends/holidays
            if (data.empty or len(data)<2) and period=="1d":
                data = yf.download(ticker, period="5d", interval=interval, progress=False)
                
            if 'Volume' in data.columns: data = data[data['Volume']>0]
            data = data.dropna()
            
            # Fetch info
            try: info = yf.Ticker(ticker).info
            except: info = {}
            try: news = yf.Ticker(ticker).news
            except: news = []
            
            # Technicals
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain/loss
            data['RSI'] = 100 - (100/(1+rs))
            data['SMA'] = data['Close'].rolling(20).mean()
            data['BB_Upper'] = data['SMA'] + 2*data['Close'].rolling(20).std()
            data['BB_Lower'] = data['SMA'] - 2*data['Close'].rolling(20).std()

            # Header
            curr_p = data['Close'].iloc[-1]
            prev_p = data['Close'].iloc[-2] if len(data)>1 else curr_p
            chg = curr_p - prev_p
            pct = (chg/prev_p)*100 if prev_p else 0
            
            # Values for Grid
            open_p = data['Open'].iloc[-1]
            high_p = data['High'].iloc[-1]
            low_p = data['Low'].iloc[-1]
            vol = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0

            # Currency
            curr_code = info.get('currency', 'USD')
            try: krw_rate = yf.Ticker("KRW=X").history(period="1d")['Close'].iloc[-1]
            except: krw_rate = 0
            price_sub = f"(₩{curr_p*krw_rate:,.0f})" if curr_code=='USD' and krw_rate else ""

            st.markdown(f"""
            <div class="finance-header">
                <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                    <div><h1 style="margin:0;">{ticker}</h1><p style="margin:0;color:#666;">{market_type}</p></div>
                    <div style="text-align:right;">
                        <h1 style="margin:0;color:{'#00C853' if chg>=0 else '#D50000'};">{curr_code} {curr_p:,.2f}</h1>
                        <p style="margin:0;font-weight:600;color:{'#00C853' if chg>=0 else '#D50000'};">{chg:+.2f} ({pct:+.2f}%) <span style="color:#888;">{price_sub}</span></p>
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

            # Tabs
            tabs = st.tabs(["Chart", "AI Analysis", "News", "Data"] + (["Fundamentals"] if market_type=="Stocks" else []))
            
            with tabs[0]:
                fig = go.Figure()
                if market_type == "Crypto":
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], fill='tozeroy', line=dict(color='#2962FF'), name='Price'))
                else:
                    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], increasing_line_color='#00C853', decreasing_line_color='#D50000'))
                
                if show_sma: fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='#FFA000', width=1), name='SMA'))
                if show_bb:
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='#999', dash='dot'), name='BB Up'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='#999', dash='dot'), name='BB Lo'))
                
                rangebreaks = [dict(bounds=["sat", "mon"])] if market_type in ["Stocks", "Commodities"] and interval in ['1m', '5m', '1h', '1d'] else []
                fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False, xaxis=dict(rangebreaks=rangebreaks))
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                # --- FEAR & GREED (Proxy) ---
                try:
                    vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
                    fear_score = max(0, min(100, 100 - (vix - 10) * 2.5))
                    fg_label = "Fear" if fear_score < 45 else "Greed" if fear_score > 55 else "Neutral"
                except: fear_score=50; fg_label="Neutral"

                # --- NEWS SENTIMENT ---
                polarities = []
                for item in news:
                    t = item.get('title')
                    if t: polarities.append(TextBlob(t).sentiment.polarity)
                avg_pol = np.mean(polarities) if polarities else 0
                news_lbl = "Positive" if avg_pol>0.05 else "Negative" if avg_pol<-0.05 else "Neutral"

                c1, c2 = st.columns([1,2])
                with c1:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=fear_score, title={'text': f"Market Sentiment"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#333"}, 'steps': [{'range': [0, 40], 'color': "#FF5252"}, {'range': [60, 100], 'color': "#00E676"}]}))
                    fig_g.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
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

                # Report Text
                trend = "Bullish" if curr_p > data['SMA'].iloc[-1] else "Bearish"
                rsi_val = data['RSI'].iloc[-1]
                report = f"### Executive Summary\n**Sentiment:** {fg_label} ({int(fear_score)}/100)\n**News:** {news_lbl}\n**Trend:** {trend}\n**RSI:** {rsi_val:.1f}"
                st.markdown(f"""<div style="background:#f8f9fa; padding:20px; border-radius:5px; border-left:4px solid #0d6efd;">{report.replace(chr(10), '<br>')}</div>""", unsafe_allow_html=True)

            with tabs[2]:
                if news:
                    for i in news[:10]:
                        t = i.get('title') or "News"
                        l = i.get('link') or "#"
                        pub = i.get('publisher', 'Source')
                        st.markdown(f"<div class='news-list-item'><div class='news-meta'>{pub}</div><a href='{l}' target='_blank' class='news-link'>{t}</a></div>", unsafe_allow_html=True)
                else: st.info("No specific news.")

            with tabs[3]:
                st.dataframe(data.tail(50), use_container_width=True)
                csv = data.to_csv().encode('utf-8')
                st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")

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

        except Exception as e: st.error(f"Error loading {ticker}: {e}")

# --- MODE: FAVORITES ---
elif mode == "Favorites":
    st.title("⭐ Watchlist")
    user_favs = db[st.session_state['user_id']]['favorites']
    if not user_favs: st.info("No favorites.")
    else:
        favs = []
        for s in user_favs:
            p, c = get_live_price(s)
            favs.append({"Ticker": s, "Price": f"${p:,.2f}", "Change": f"{c:+.2f}%"})
        st.dataframe(pd.DataFrame(favs), use_container_width=True)

# --- MODE: MEDIA ---
elif mode == "Media & News":
    st.title("📺 Media Center")
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
    with t2: render_feed("http://feeds.bbci.co.uk/news/business/rss.xml")
    with t3: render_feed("http://rss.cnn.com/rss/money_latest.rss")
