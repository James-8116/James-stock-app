import os
import time
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 網頁基本設定
st.set_page_config(
    page_title="台股 AI 逆勢波段選股系統",
    page_icon="📉",
    layout="wide"
)

# 優先讀取 Streamlit Secrets，若無則讀取環境變數
FINMIND_TOKEN = st.secrets.get("FINMIND_TOKEN", os.environ.get("FINMIND_TOKEN", ""))
FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=86400)
def get_tw_stock_list(token):
    """快取 24 小時：從 FinMind 取得台股上市櫃一般股票清單"""
    params = {"dataset": "TaiwanStockInfo", "token": token}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=15)
        data = res.json()
        if data.get("msg") != "success":
            return {}
        df = pd.DataFrame(data["data"])
        df = df[df["type"].isin(["twse", "tpex"])]
        df = df[df["stock_id"].str.len() == 4]
        df = df[df["stock_id"].str.isdigit()]
        stock_map = {}
        for _, row in df.iterrows():
            suffix = ".TW" if row["type"] == "twse" else ".TWO"
            stock_map[row["stock_id"] + suffix] = {
                "id": row["stock_id"],
                "name": row["stock_name"],
                "market": "上市" if row["type"] == "twse" else "上櫃"
            }
        return stock_map
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def download_all_market_data(tickers):
    """快取 1 小時：批次下載全市場近 4 個月日K線，調整參數時免重複下載"""
    all_dfs = {}
    batch_size = 250
    progress_bar = st.progress(0, text="正在從 Yahoo Finance 下載全市場日 K 線...")

    total_batches = (len(tickers) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
            for ticker in batch:
                if ticker in data.columns.get_level_values(0):
                    df = data[ticker].dropna(subset=["Close", "Volume"])
                    if len(df) >= 62:
                        all_dfs[ticker] = df
        except Exception:
            pass
        progress_bar.progress((idx + 1) / total_batches, text=f"下載進度：{idx + 1} / {total_batches} 批次")
        time.sleep(0.5)

    progress_bar.empty()
    return all_dfs

def check_inst_buy_5d(stock_id, token):
    """查詢單一股票近 5 日三大法人合計買賣超 (張)"""
    start_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date,
        "token": token
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10)
        data = res.json()
        if data.get("msg") != "success" or not data.get("data"):
            return 0
        df = pd.DataFrame(data["data"])
        df["net"] = df["buy"] - df["sell"]
        recent_dates = sorted(df["date"].unique())[-5:]
        return df[df["date"].isin(recent_dates)]["net"].sum() / 1000
    except Exception:
        return 0

# --- 側邊欄參數設定 ---
with st.sidebar:
    st.header("⚙️ 選股參數微調")
    st.caption("預設參數與每日 17:00 LINE 推播機器人完全同步")

    rsi_threshold = st.slider("14日 RSI 超賣上限（越低越超賣）", min_value=25, max_value=55, value=40, step=1)
    min_vol_lots = st.number_input("5日平均成交量下限（張）", min_value=300, max_value=10000, value=1000, step=100)
    min_price = st.number_input("最低股價門檻（元）", min_value=10, max_value=200, value=15, step=5)
    require_inst_buy = st.checkbox("強制要求近 5 日三大法人買超 > 0", value=True)
    top_n = st.slider("顯示精選檔數", min_value=3, max_value=15, value=5)

    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")

    run_btn = st.button("🚀 開始全市場掃描", type="primary", use_container_width=True)

# --- 主畫面 ---
st.title("📉 台股逆勢波段選股看板（下跌找買點・上漲找賣點）")
st.markdown("策略邏輯：**60日季線之上（多頭保護）** ＋ **跌破20日月線（短線拉回）** ＋ **RSI 超賣止跌勾頭** ＋ **法人低接**")

if not FINMIND_TOKEN:
    st.warning("⚠️ 尚未偵測到 `FINMIND_TOKEN`，請在 Streamlit Cloud 的 Secrets 設定中新增，或直接在左側邊欄輸入 Token。")
    st.stop()

if run_btn or "scan_results" in st.session_state:
    stock_map = get_tw_stock_list(FINMIND_TOKEN)
    if not stock_map:
        st.error("無法取得台股清單，請檢查 FinMind Token 是否正確。")
        st.stop()

    market_data = download_all_market_data(list(stock_map.keys()))

    technical_passed = []
    for ticker, df in market_data.items():
        close = df["Close"]
        open_p = df["Open"]
        volume = df["Volume"] / 1000

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol_ma5 = volume.rolling(5).mean()
        rsi14 = calculate_rsi(close, 14)

        curr_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        curr_open = float(open_p.iloc[-1])
        curr_ma20 = float(ma20.iloc[-1])
        curr_ma60 = float(ma60.iloc[-1])
        curr_vol_ma5 = float(vol_ma5.iloc[-1])
        curr_rsi = float(rsi14.iloc[-1])
        prev_rsi = float(rsi14.iloc[-2])
        high_20d = float(df["High"].iloc[-20:].max())

        if curr_vol_ma5 < min_vol_lots or curr_close < min_price:
            continue
        if curr_close <= curr_ma60:
            continue
        if curr_close >= curr_ma20:
            continue

        is_rebounding = (curr_rsi >= prev_rsi) or (curr_close > curr_open)
        if not (curr_rsi < rsi_threshold and is_rebounding):
            continue

        info = stock_map[ticker]
        technical_passed.append({
            "ticker": ticker,
            "stock_id": info["id"],
            "name": info["name"],
            "market": info["market"],
            "close": curr_close,
            "pct": ((curr_close - prev_close) / prev_close) * 100,
            "rsi": curr_rsi,
            "ma20": curr_ma20,
            "ma60": curr_ma60,
            "high_20d": high_20d,
            "upside_ma20": ((curr_ma20 - curr_close) / curr_close) * 100,
            "upside_high": ((high_20d - curr_close) / curr_close) * 100,
            "vol_ma5": int(curr_vol_ma5)
        })

    # 查詢法人籌碼
    final_list = []
    with st.spinner(f"技術面共 {len(technical_passed)} 檔過關，正在核對三大法人買賣超..."):
        for item in technical_passed:
            net_5d = check_inst_buy_5d(item["stock_id"], FINMIND_TOKEN)
            if require_inst_buy and net_5d <= 0:
                continue
            item["net_5d"] = int(net_5d)
            final_list.append(item)

    if not final_list:
        st.info("💡 目前參數條件下無符合標的。你可以嘗試在左側邊欄將 **「14日 RSI 超賣上限」調高至 45～50**，或暫時取消勾選法人買超限制，查看最接近買點的潛力股。")
    else:
        df_res = pd.DataFrame(final_list)
        df_res = df_res.sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(top_n)
        st.session_state["scan_results"] = df_res

        st.subheader(f"🎯 精選波段買點標的（共 {len(df_res)} 檔）")

        # 整理漂亮表格顯示
        display_df = pd.DataFrame({
            "代號": df_res["stock_id"],
            "名稱": df_res["name"],
            "市場": df_res["market"],
            "現價": df_res["close"].map(lambda x: f"{x:.1f}"),
            "漲跌幅": df_res["pct"].map(lambda x: f"{x:+.2f}%"),
            "14日 RSI": df_res["rsi"].map(lambda x: f"{x:.1f}"),
            "5日法人(張)": df_res["net_5d"].map(lambda x: f"{x:+,}"),
            "🟢 第一賣點(月線)": df_res.apply(lambda r: f"{r['ma20']:.1f} (+{r['upside_ma20']:.1f}%)", axis=1),
            "🔴 第二賣點(前高)": df_res.apply(lambda r: f"{r['high_20d']:.1f} (+{r['upside_high']:.1f}%)", axis=1),
            "🛡️ 防守停損(季線)": df_res["ma60"].map(lambda x: f"{x:.1f}"),
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # --- 個股均線與買賣點視覺化預覽 ---
        st.divider()
        st.subheader("📈 個股均線與買賣點區間預覽")
        options = [f"{r['stock_id']} {r['name']}（現價 {r['close']:.1f} / RSI {r['rsi']:.1f}）" for _, r in df_res.iterrows()]
        selected_option = st.selectbox("選擇要檢視走勢的標的：", options)
        selected_idx = options.index(selected_option)
        sel_row = df_res.iloc[selected_idx]

        chart_df = market_data[sel_row["ticker"]].copy()
        chart_df["20日月線 (第一賣點)"] = chart_df["Close"].rolling(20).mean()
        chart_df["60日季線 (防守停損)"] = chart_df["Close"].rolling(60).mean()
        chart_df = chart_df[["Close", "20日月線 (第一賣點)", "60日季線 (防守停損)"]].tail(60)
        chart_df.columns = ["收盤價", "20日月線 (第一賣點)", "60日季線 (防守停損)"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盤價", f"{sel_row['close']:.1f}", f"{sel_row['pct']:+.2f}%")
        col2.metric("第一賣點 (月線)", f"{sel_row['ma20']:.1f}", f"+{sel_row['upside_ma20']:.1f}%")
        col3.metric("第二賣點 (前波高)", f"{sel_row['high_20d']:.1f}", f"+{sel_row['upside_high']:.1f}%")
        col4.metric("防守停損 (季線)", f"{sel_row['ma60']:.1f}", f"{((sel_row['ma60']-sel_row['close'])/sel_row['close'])*100:.1f}%")

        st.line_chart(chart_df, color=["#FF4B4B", "#FFA500", "#00CC96"])
