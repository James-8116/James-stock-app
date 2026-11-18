import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# 設定手機版面適應
st.set_page_config(page_title="AI 波浪黃金分割決策 APP", layout="centered")

st.title("📈 AI 波浪與黃金分割決策 APP")
st.caption("整合：波段高低點 + 費波那契回檔 + RSI動能 + 量能確認 + ATR動態風控")

# 1. 載入全市場台股代號與中文股名對照表（每日自動更新並快取）
@st.cache_data(ttl=86400)
def load_stock_mapping():
    code_map = {}  # 代號 -> (yf_ticker, 完整顯示名稱)
    name_map = {}  # 股名 -> (yf_ticker, 完整顯示名稱)
    
    # 抓取上市清單 (TWSE)
    try:
        twse_url = "https://twse.com.tw"
        for item in requests.get(twse_url, timeout=10).json():
            code = item.get("Code", "").strip()
            name = item.get("Name", "").strip()
            if code and name:
                ticker = f"{code}.TW"
                display = f"{code} {name}"
                code_map[code] = (ticker, display)
                name_map[name] = (ticker, display)
    except Exception:
        pass

    # 抓取上櫃清單 (TPEx)
    try:
        tpex_url = "https://tpex.org.tw"
        for item in requests.get(tpex_url, timeout=10).json():
            code = item.get("SecuritiesCompanyCode", "").strip()
            name = item.get("CompanyName", "").strip()
            if code and name:
                ticker = f"{code}.TWO"
                display = f"{code} {name}"
                code_map[code] = (ticker, display)
                name_map[name] = (ticker, display)
    except Exception:
        pass

    return code_map, name_map

# 2. 智慧解析使用者輸入（支援代號、完整股名、部分關鍵字）
def resolve_stock_query(query, code_map, name_map):
    q = query.strip().upper()
    if not q:
        return None, None
        
    # 完全符合代號
    if q in code_map:
        return code_map[q]
        
    # 完全符合中文股名
    if q in name_map:
        return name_map[q]
        
    # 模糊比對中文股名（例如輸入「台積」自動找到「台積電」）
    matches = [info for name, info in name_map.items() if q in name]
    if matches:
        # 優先回傳名稱最短（最接近）的標的
        matches.sort(key=lambda x: len(x[1]))
        return matches[0]
        
    # 若對照表暫時連線異常，保留直接嘗試代號的備援機制
    return None, q

# 3. 抓取台股歷史數據函式
@st.cache_data(ttl=1800)
def get_stock_data(resolved_ticker, fallback_code):
    if resolved_ticker:
        df = yf.download(resolved_ticker, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, resolved_ticker
            
    # 備援嘗試
    for suffix in [".TWO", ".TW"]:
        df = yf.download(f"{fallback_code}{suffix}", period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, f"{fallback_code}{suffix}"
    return pd.DataFrame(), None

# 4. 核心指標與買賣區間運算引擎
def analyze_stock(df, days):
    sub_df = df.tail(days).copy()
    current_price = float(sub_df['Close'].iloc[-1])
    
    swing_high = float(sub_df['High'].max())
    swing_low = float(sub_df['Low'].min())
    diff = swing_high - swing_low
    
    fib_382 = swing_high - diff * 0.382
    fib_500 = swing_high - diff * 0.500
    fib_618 = swing_high - diff * 0.618
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    bias20 = ((current_price - ma20) / ma20) * 100
    
    vol_today = float(df['Volume'].iloc[-1])
    vol_ma5 = float(df['Volume'].rolling(5).mean().iloc[-1])
    vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0
    
    buy_high = max(fib_618, current_price) if current_price <= fib_618 * 1.03 else fib_618
    buy_low = max(swing_low, fib_618 - 0.5 * atr)
    stop_loss = swing_low - (0.5 * atr)
    
    tp1 = fib_382
    tp2 = swing_high - diff * 0.236 if fib_500 <= current_price else fib_500
    
    avg_buy = (buy_high + buy_low) / 2
    risk = avg_buy - stop_loss
    reward = tp1 - avg_buy
    rr_ratio = reward / risk if risk > 0 else 0
    
    return {
        "price": current_price, "high": swing_high, "low": swing_low,
        "fib_382": fib_382, "fib_500": fib_500, "fib_618": fib_618,
        "rsi": rsi, "atr": atr, "bias20": bias20, "vol_ratio": vol_ratio,
        "buy_low": buy_low, "buy_high": buy_high,
        "stop_loss": stop_loss, "tp1": tp1, "tp2": tp2, "rr_ratio": rr_ratio
    }

# 5. 使用者介面互動區
code_map, name_map = load_stock_mapping()

user_query = st.text_input("輸入台股代號或股名（例如：4908、前鼎、聯亞）", value="4908")
lookback_days = st.slider("波段觀察天數（決定高低點基準）", min_value=30, max_value=240, value=90)
rr_threshold = st.number_input("最低要求風險報酬比（賺賠比）", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

if st.button("🚀 啟動 AI 量化診斷", use_container_width=True):
    with st.spinner("正在配對個股並運算波浪結構與波動率指標..."):
        resolved_ticker, display_title = resolve_stock_query(user_query, code_map, name_map)
        df, final_ticker = get_stock_data(resolved_ticker, user_query.strip())
        
        if df.empty:
            st.error(f"找不到「{user_query}」的相關資料，請確認輸入的代號或中文股名是否正確。")
        else:
            res = analyze_stock(df, lookback_days)
            title_text = display_title if resolved_ticker else final_ticker
            
            st.subheader(f"📊 {title_text} 量化交易計畫")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("最新收盤價", f"{res['price']:.1f} 元", f"月線乖離 {res['bias20']:.1f}%")
            col2.metric("14日 RSI", f"{res['rsi']:.1f}", "低檔轉折區" if res['rsi'] < 42 else "中性/偏強")
            col3.metric("今日量能比", f"{res['vol_ratio']:.2f} 倍", "爆量換手" if res['vol_ratio'] >= 1.5 else "量縮整理")
            
            st.divider()
            st.markdown("### 🎯 具體買賣區間與停損停利點")
            
            c1, c2 = st.columns(2)
            c1.success(f"🟢 **建議分批買進區間**\n\n**{res['buy_low']:.1f} ～ {res['buy_high']:.1f} 元**")
            c2.error(f"🔴 **防洗盤嚴格停損價**\n\n**{res['stop_loss']:.1f} 元**（低點 - 0.5 ATR）")
            
            c3, c4 = st.columns(2)
            c3.info(f"🎯 **第一停利位 (0.382)**\n\n**{res['tp1']:.1f} 元**（建議減碼一半）")
            c4.info(f"🏆 **第二停利位 (0.500)**\n\n**{res['tp2']:.1f} 元**（波段全數出場）")
            
            st.write(f"**波段基準區間：** 最高 `{res['high']:.1f} 元` ｜ 最低 `{res['low']:.1f} 元` ｜ **預估賺賠比：** `{res['rr_ratio']:.2f} 倍`")
            
            if res['rr_ratio'] >= rr_threshold and res['rsi'] < 42 and res['price'] <= res['fib_618'] * 1.03:
                st.success("✅ **【強烈關注買進】** 股價已回測 0.618 黃金支撐區，RSI 處於低檔且賺賠比大於設定門檻！")
            elif res['rr_ratio'] < rr_threshold:
                st.warning(f"⚠️ **【建議觀望】** 目前距離停損點過遠或上方反彈空間不足，賺賠比（{res['rr_ratio']:.2f}）未達 {rr_threshold} 倍標準。")
            else:
                st.info("ℹ️ **【等待回測】** 股價尚未落入 0.618 黃金買進區或 RSI 尚未超賣，建議耐心等待拉回買點。")
            
            st.divider()
            st.markdown("### 📉 走勢與黃金分割關鍵防線圖")
            
            chart_df = pd.DataFrame(index=df.tail(lookback_days).index)
            chart_df['收盤價'] = df['Close'].tail(lookback_days)
            chart_df['第一停利 (0.382)'] = res['tp1']
            chart_df['黃金買進線 (0.618)'] = res['fib_618']
            chart_df['嚴格停損線'] = res['stop_loss']
            
            st.line_chart(chart_df, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
