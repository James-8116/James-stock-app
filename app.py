import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 設定手機版面適應
st.set_page_config(page_title="AI 波浪黃金分割選股儀", layout="centered")

st.title("📈 AI 波浪與黃金分割決策 APP")
st.caption("整合：波段高低點 + 費波那契回檔 + RSI動能 + ATR動態停損 + 賺賠比檢核")

# 1. 使用者輸入區（支援手機觸控輸入）
stock_id = st.text_input("輸入台股代號（例如：4908 或 3081）", value="4908")
lookback_days = st.slider("波段觀察天數（決定高低點基準）", min_value=30, max_value=240, value=90)
rr_threshold = st.number_input("最低要求風險報酬比（賺賠比）", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

# 2. 抓取台股數據函式
@st.cache_data(ttl=3600)
def get_stock_data(ticker_code):
    # 嘗試上市 (.TW) 與上櫃 (.TWO)
    for suffix in [".TWO", ".TW"]:
        df = yf.download(f"{ticker_code}{suffix}", period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, f"{ticker_code}{suffix}"
    return pd.DataFrame(), None

# 3. 核心指標運算引擎
def analyze_stock(df, days):
    sub_df = df.tail(days).copy()
    current_price = float(sub_df['Close'].iloc[-1])
    
    # 計算波段最高與最低點
    swing_high = float(sub_df['High'].max())
    swing_low = float(sub_df['Low'].min())
    diff = swing_high - swing_low
    
    # 黃金分割位階計算（自高點回檔）
    fib_382 = swing_high - diff * 0.382
    fib_500 = swing_high - diff * 0.500
    fib_618 = swing_high - diff * 0.618
    
    # 計算 14 日 RSI 動能
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    
    # 計算 14 日 ATR（真實波動幅度）設定動態停損
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    
    # 計算 20 日均線與負乖離率 (BIAS)
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    bias20 = ((current_price - ma20) / ma20) * 100
    
    # 買賣點與風控試算（假設以 0.618 或現價評估反彈至 0.382）
    entry_price = current_price
    stop_loss = swing_low - (0.5 * atr)  # 波段低點下方 0.5 倍 ATR 作為防洗盤停損
    take_profit = fib_382 if current_price < fib_382 else swing_high
    
    risk = entry_price - stop_loss
    reward = take_profit - entry_price
    rr_ratio = reward / risk if risk > 0 else 0
    
    return {
        "price": current_price, "high": swing_high, "low": swing_low,
        "fib_382": fib_382, "fib_500": fib_500, "fib_618": fib_618,
        "rsi": rsi, "atr": atr, "bias20": bias20,
        "stop_loss": stop_loss, "take_profit": take_profit, "rr_ratio": rr_ratio
    }

# 4. 執行分析與手機介面呈現
if st.button("🚀 啟動 AI 量化診斷", use_container_width=True):
    with st.spinner("正在運算波浪結構與波動率指標..."):
        df, full_ticker = get_stock_data(stock_id.strip())
        
        if df.empty:
            st.error("找不到該股票代號，請確認輸入是否正確（例如：4908、3081、2330）。")
        else:
            res = analyze_stock(df, lookback_days)
            
            st.subheader(f"📊 {full_ticker} 診斷結果")
            
            # 關鍵數據卡片（手機雙欄排版）
            col1, col2 = st.columns(2)
            col1.metric("最新收盤價", f"{res['price']:.2f} 元", f"月線乖離 {res['bias20']:.1f}%")
            col2.metric("14日 RSI 動能", f"{res['rsi']:.1f}", "超賣區 (<35)" if res['rsi'] < 35 else "中性/偏強")
            
            st.divider()
            st.markdown("### 🎯 黃金分割關鍵點位")
            st.write(f"- **波段高點：** {res['high']:.2f} 元 ｜ **波段低點：** {res['low']:.2f} 元")
            st.write(f"- **0.382 弱勢反彈位（第一停利）：** `{res['fib_382']:.2f} 元`")
            st.write(f"- **0.500 中軸平衡位：** `{res['fib_500']:.2f} 元`")
            st.write(f"- **0.618 黃金買進支撐區：** `{res['fib_618']:.2f} 元`")
            
            st.divider()
            st.markdown("### 🛡️ ATR 動態風控與交易評級")
            st.write(f"- **建議防洗盤停損價（低點 - 0.5 ATR）：** `{res['stop_loss']:.2f} 元`")
            st.write(f"- **預估風險報酬比（賺賠比）：** `{res['rr_ratio']:.2f} 倍`")
            
            # 綜合 AI 評級邏輯
            if res['rr_ratio'] >= rr_threshold and res['rsi'] < 45:
                st.success("✅ **【符合買進標準】** 目前位於波段低檔支撐區，RSI 動能偏低具反彈空間，且賺賠比大於設定門檻！")
            elif res['rr_ratio'] < rr_threshold:
                st.warning(f"⚠️ **【建議觀望】** 目前距離停損點過遠或上方空間不足，賺賠比（{res['rr_ratio']:.2f}）未達 {rr_threshold} 倍標準。")
            else:
                st.info("ℹ️ **【中性追蹤】** 股價尚未落入極端超賣區，建議等待回測 0.618 支撐位再行介入。")
                
            # 繪製近期走勢圖
            st.line_chart(df['Close'].tail(lookback_days))
