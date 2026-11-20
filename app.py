import os
import time
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="台股 AI 逆勢波段選股", page_icon="📉", layout="wide")

FINMIND_TOKEN = st.secrets.get("FINMIND_TOKEN", os.environ.get("FINMIND_TOKEN", ""))
FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def check_fib_touch(curr_close, curr_low, high_60d, low_60d, tol=0.015):
    swing = high_60d - low_60d
    if swing <= 0:
        return None, 0.0, {}
    fibs = {
        "0.382": high_60d - swing * 0.382,
        "0.500": high_60d - swing * 0.500,
        "0.618": high_60d - swing * 0.618,
    }
    for k, v in fibs.items():
        if abs(curr_close - v) / v <= tol or abs(curr_low - v) / v <= tol:
            return k, v, fibs
    return None, 0.0, fibs

@st.cache_data(ttl=86400)
def get_tw_stock_list(token):
    try:
        res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo", "token": token}, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        df = df[df["type"].isin(["twse", "tpex"]) & (df["stock_id"].str.len() == 4) & (df["stock_id"].str.isdigit())]
        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {"id": r["stock_id"], "name": r["stock_name"]}
            for _, r in df.iterrows()
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def download_all_market_data(tickers):
    all_dfs = {}
    batch_size = 250
    bar = st.progress(0, text="下載全市場日 K 線中...")
    total = (len(tickers) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t in data.columns.get_level_values(0):
                    df = data[t].dropna(subset=["Close", "Volume"])
                    if len(df) >= 62:
                        all_dfs[t] = df
        except Exception:
            pass
        bar.progress((idx + 1) / total, text=f"下載進度：{idx + 1} / {total}")
        time.sleep(0.5)
    bar.empty()
    return all_dfs

def check_inst_buy_5d(stock_id, token):
    start = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start, "token": token}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10).json()
        if res.get("msg") != "success" or not res.get("data"):
            return 0
        df = pd.DataFrame(res["data"])
        df["net"] = df["buy"] - df["sell"]
        recent = sorted(df["date"].unique())[-5:]
        return df[df["date"].isin(recent)]["net"].sum() / 1000
    except Exception:
        return 0

with st.sidebar:
    st.header("⚙️ 選股參數微調")
    rsi_limit = st.slider("14日 RSI 超賣上限", 25, 60, 48, 1)
    min_pullback = st.slider("距20日高點至少回檔 (%)", 1, 20, 3, 1)
    min_vol = st.number_input("5日均量下限 (張)", 300, 10000, 1000, 100)
    min_price = st.number_input("最低股價 (元)", 10, 200, 15, 5)
    st.divider()
    req_fib = st.checkbox("強制碰觸黃金分割 (0.382/0.5/0.618)", value=True)
    fib_tol = st.slider("黃金分割誤差範圍 (±%)", 0.5, 3.0, 1.5, 0.5) / 100
    req_ma20 = st.checkbox("必須跌破 20 日月線", value=False)
    req_rebound = st.checkbox("必須當日止跌 (收紅K或RSI勾頭)", value=False)
    req_inst = st.checkbox("強制近 5 日三大法人買超 > 0", value=True)
    top_n = st.slider("顯示精選檔數", 3, 15, 5)
    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")
    run_btn = st.button("🚀 開始全市場掃描", type="primary", use_container_width=True)

st.title("📉 台股逆勢波段選股看板（黃金分割＋法人低接）")
st.markdown("策略：**60日季線之上（多頭保護）** ＋ **回測黃金分割支撐** ＋ **RSI 低檔** ＋ **法人低接**")

if not FINMIND_TOKEN:
    st.warning("⚠️ 請在左側邊欄輸入 FinMind Token 或設定 Streamlit Secrets。")
    st.stop()

if run_btn or "market_cached" in st.session_state:
    stock_map = get_tw_stock_list(FINMIND_TOKEN)
    if not stock_map:
        st.error("無法取得台股清單，請檢查 Token。")
        st.stop()

    market_data = download_all_market_data(list(stock_map.keys()))
    st.session_state["market_cached"] = True

    passed = []
    for t, df in market_data.items():
        close, open_p, low_p = df["Close"], df["Open"], df["Low"]
        vol_ma5 = (df["Volume"] / 1000).rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        rsi = calculate_rsi(close, 14)
        curr_c, prev_c, curr_o, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(open_p.iloc[-1]), float(low_p.iloc[-1])
        curr_rsi, prev_rsi = float(rsi.iloc[-1]), float(rsi.iloc[-2])
        high_20d = float(df["High"].iloc[-20:].max())
        high_60d, low_60d = float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())

        if vol_ma5 < min_vol or curr_c < min_price or curr_c <= ma60:
            continue
        pullback = ((high_20d - curr_c) / high_20d) * 100
        if pullback < min_pullback:
            continue
        fib_hit, fib_p, fib_dict = check_fib_touch(curr_c, curr_l, high_60d, low_60d, fib_tol)
        if req_fib and not fib_hit:
            continue
        if req_ma20 and curr_c >= ma20:
            continue
        if curr_rsi >= rsi_limit:
            continue
        if req_rebound and not (curr_rsi >= prev_rsi or curr_c > curr_o):
            continue

        info = stock_map[t]
        passed.append({
            "ticker": t, "stock_id": info["id"], "name": info["name"],
            "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
            "rsi": curr_rsi, "fib_hit": fib_hit or "未觸及", "fib_price": fib_p, "fib_dict": fib_dict,
            "ma20": ma20, "ma60": ma60, "high_20d": high_20d,
            "up_ma20": ((ma20 - curr_c) / curr_c) * 100, "up_high": ((high_20d - curr_c) / curr_c) * 100
        })

    passed = sorted(passed, key=lambda x: x["rsi"])[:40]
    final_list = []
    with st.spinner(f"技術面 {len(passed)} 檔過關，核對三大法人買賣超中..."):
        for item in passed:
            net = check_inst_buy_5d(item["stock_id"], FINMIND_TOKEN)
            if req_inst and net <= 0:
                continue
            item["net_5d"] = int(net)
            final_list.append(item)

    if not final_list:
        st.info("💡 目前條件無符合標的，請在左側放寬 RSI 上限或黃金分割誤差範圍。")
    else:
        df_res = pd.DataFrame(final_list).sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(top_n)
        st.subheader(f"🎯 精選黃金分割波段買點（共 {len(df_res)} 檔）")

        disp = pd.DataFrame({
            "代號": df_res["stock_id"],
            "名稱": df_res["name"],
            "現價": df_res["close"].map(lambda x: f"{x:.1f}"),
            "漲跌幅": df_res["pct"].map(lambda x: f"{x:+.2f}%"),
            "📐 黃金分割支撐": df_res.apply(lambda r: f"{r['fib_hit']} ({r['fib_price']:.1f})" if r['fib_hit'] != "未觸及" else "未觸及", axis=1),
            "14日 RSI": df_res["rsi"].map(lambda x: f"{x:.1f}"),
            "5日法人(張)": df_res["net_5d"].map(lambda x: f"{x:+,}"),
            "🟢 第一賣點(月線)": df_res.apply(lambda r: f"{r['ma20']:.1f} ({r['up_ma20']:+.1f}%)", axis=1),
            "🔴 第二賣點(前高)": df_res.apply(lambda r: f"{r['high_20d']:.1f} (+{r['up_high']:.1f}%)", axis=1),
            "🛡️ 防守停損(季線)": df_res["ma60"].map(lambda x: f"{x:.1f}"),
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📈 個股均線與黃金分割支撐預覽")
        opts = [f"{r['stock_id']} {r['name']}（現價 {r['close']:.1f} / 碰觸 {r['fib_hit']}）" for _, r in df_res.iterrows()]
        sel_row = df_res.iloc[opts.index(st.selectbox("選擇檢視標的：", opts))]

        cdf = market_data[sel_row["ticker"]].copy()
        fibs = sel_row["fib_dict"]
        cdf["20日月線"] = cdf["Close"].rolling(20).mean()
        cdf["60日季線"] = cdf["Close"].rolling(60).mean()
        cdf["黃金 0.382"], cdf["黃金 0.500"], cdf["黃金 0.618"] = fibs["0.382"], fibs["0.500"], fibs["0.618"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤價", f"{sel_row['close']:.1f}", f"{sel_row['pct']:+.2f}%")
        c2.metric(f"黃金分割 ({sel_row['fib_hit']})", f"{sel_row['fib_price']:.1f}" if sel_row['fib_price'] > 0 else "無", f"0.618位: {fibs['0.618']:.1f}")
        c3.metric("反彈賣點 (前波高)", f"{sel_row['high_20d']:.1f}", f"+{sel_row['up_high']:.1f}%")
        c4.metric("防守停損 (季線)", f"{sel_row['ma60']:.1f}", f"{((sel_row['ma60']-sel_row['close'])/sel_row['close'])*100:.1f}%")

        plot_df = cdf[["Close", "20日月線", "60日季線", "黃金 0.382", "黃金 0.500", "黃金 0.618"]].tail(60)
        plot_df.columns = ["收盤價", "20日月線", "60日季線", "黃金 0.382", "黃金 0.500", "黃金 0.618"]
        st.line_chart(plot_df, color=["#FF4B4B", "#FFA500", "#00CC96", "#636EFA", "#AB63FA", "#19D3F3"])
