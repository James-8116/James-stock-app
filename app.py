import os
import time
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="台股 AI 波段監控與選股", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] div {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)

FINMIND_TOKEN = st.secrets.get("FINMIND_TOKEN", os.environ.get("FINMIND_TOKEN", ""))
FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"
TWSE_T86_URL = "https://openapi." + "twse" + ".com.tw/v1/fund/T86"
TW_TZ = timezone(timedelta(hours=8))

DEFAULT_HOLDINGS = [
    {"代號": "2891", "買進成本": 71.4},
    {"代號": "5876", "買進成本": 50.9},
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def check_fib_touch(curr_c, curr_l, h60, l60, ma60, tol=0.015):
    swing = h60 - l60
    if swing <= 0:
        return None, 0.0, {}, False
    fibs = {"0.382": h60 - swing * 0.382, "0.500": h60 - swing * 0.500, "0.618": h60 - swing * 0.618}
    for k, v in fibs.items():
        if abs(curr_c - v) / v <= tol or abs(curr_l - v) / v <= tol:
            is_conf = abs(v - ma60) / ma60 <= 0.025
            return k, v, fibs, is_conf
    return None, 0.0, fibs, False

@st.cache_data(ttl=55)
def get_twii_status(lookback_days=60):
    try:
        df = yf.download("^TWII", period="18mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            return None
        close = df["Close"]
        curr_c, prev_c = float(close.iloc[-1]), float(close.iloc[-2])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma240 = float(close.rolling(240).mean().iloc[-1]) if len(close) >= 240 else float(close.mean())
        rsi_s = calculate_rsi(close, 14)
        rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
        sub = df.iloc[-lookback_days:]
        h_val, l_val = float(sub["High"].max()), float(sub["Low"].min())
        swing = h_val - l_val
        return {
            "close": curr_c, "diff": curr_c - prev_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
            "ma60": ma60, "ma240": ma240, "rsi": rsi, "prev_rsi": prev_rsi,
            "fib382": h_val - swing * 0.382,
            "fib500": h_val - swing * 0.500,
            "fib618": h_val - swing * 0.618,
            "fib_ext1272": max(0.0, h_val - swing * 1.272)
        }
    except Exception:
        return None
@st.cache_data(ttl=86400)
def get_all_tw_names(token):
    try:
        res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo", "token": token}, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        return dict(zip(df["stock_id"].astype(str), df["stock_name"]))
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def get_tw_stock_list(token):
    try:
        res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo", "token": token}, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        df = df[df["type"].isin(["twse", "tpex"]) & (df["stock_id"].str.len() == 4) & (df["stock_id"].str.isdigit())]
        return {r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {"id": r["stock_id"], "name": r["stock_name"]} for _, r in df.iterrows()}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_top5_institutional(token):
    stock_map = get_tw_stock_list(token)
    id_to_name = {v["id"]: v["name"] for v in stock_map.values()}
    if not id_to_name:
        return pd.DataFrame(), pd.DataFrame(), ""
    net_dict, data_date = {}, ""
    for days_back in range(0, 5):
        q_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "start_date": q_date, "end_date": q_date, "token": token}, timeout=15).json()
            if res.get("msg") == "success" and res.get("data"):
                df = pd.DataFrame(res["data"])
                df = df[df["stock_id"].isin(id_to_name.keys())]
                if not df.empty:
                    df["net_lots"] = (df["buy"] - df["sell"]) / 1000
                    net_dict = df.groupby("stock_id")["net_lots"].sum().to_dict()
                    data_date = q_date
                    break
        except Exception:
            pass
    if not net_dict:
        try:
            rows = requests.get(TWSE_T86_URL, timeout=15).json()
            for r in rows:
                sid = str(r.get("Code", "")).strip()
                if sid in id_to_name:
                    val = str(r.get("TotalInstitutionalInvestorsNetBuySell", "0")).replace(",", "")
                    net_dict[sid] = float(val) / 1000
            data_date = "最新交易日"
        except Exception:
            pass
    if not net_dict:
        return pd.DataFrame(), pd.DataFrame(), ""
    items = [{"代號": k, "名稱": id_to_name[k], "買賣超(張)": int(round(v))} for k, v in net_dict.items()]
    buy_df = pd.DataFrame(sorted([x for x in items if x["買賣超(張)"] > 0], key=lambda x: x["買賣超(張)"], reverse=True)[:5])
    sell_df = pd.DataFrame(sorted([x for x in items if x["買賣超(張)"] < 0], key=lambda x: x["買賣超(張)"])[:5])
    if not buy_df.empty:
        buy_df["買賣超(張)"] = buy_df["買賣超(張)"].map(lambda x: f"{x:+,}")
    if not sell_df.empty:
        sell_df["買賣超(張)"] = sell_df["買賣超(張)"].map(lambda x: f"{x:+,}")
    return buy_df, sell_df, data_date

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

@st.cache_data(ttl=55)
def fetch_single_stock(code):
    ticker_str = code + ".TW"
    df = yf.download(ticker_str, period="4mo", progress=False)
    if df.empty:
        ticker_str = code + ".TWO"
        df = yf.download(ticker_str, period="4mo", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cn_names = get_all_tw_names(FINMIND_TOKEN)
    stock_name = cn_names.get(code)
    if not stock_name:
        try:
            info = yf.Ticker(ticker_str).info
            stock_name = info.get("shortName") or info.get("longName") or code
        except Exception:
            stock_name = code
    return df, stock_name

with st.sidebar:
    st.header("⏱️ 盤中即時更新設定")
    auto_refresh = st.toggle("🟢 開啟盤中即時自動更新 (60秒)", value=True)
    refresh_sec = 60 if auto_refresh else None
    st.divider()
    st.header("⚙️ 全市場選股參數微調")
    rsi_limit = st.slider("14日 RSI 超賣上限", 25, 60, 48, 1)
    min_pullback = st.slider("距20日高點至少回檔 (%)", 1, 20, 3, 1)
    min_vol = st.number_input("5日均量下限 (張)", 300, 10000, 1000, 100)
    min_price = st.number_input("最低股價 (元)", 10, 200, 15, 5)
    st.divider()
    req_fib = st.checkbox("強制碰觸黃金分割 (0.382/0.5/0.618)", value=True)
    req_conf_only = st.checkbox("🔥 只看「黃金分割＋季線重疊」雙支撐共振股", value=False)
    fib_tol = st.slider("黃金分割誤差範圍 (±%)", 0.5, 3.0, 1.5, 0.5) / 100
    req_ma20 = st.checkbox("必須跌破 20 日月線", value=False)
    req_rebound = st.checkbox("必須當日止跌 (收紅K或RSI勾頭)", value=False)
    req_inst = st.checkbox("強制近 5 日三大法人買超 > 0", value=True)
    top_n = st.slider("顯示精選檔數", 3, 15, 5)
    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")

st.title("📊 台股 AI 波段監控與選股看板")

@st.fragment(run_every=refresh_sec)
def render_live_market_panel():
    now_tw = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    status_txt = "🟢 即時自動更新中 (每60秒)" if auto_refresh else "⏸️ 已暫停自動更新"
    st.caption(f"🕒 最後報價更新：{now_tw}（台灣時間）｜ {status_txt}")

    period_map = {"60日 (一季波段)": 60, "120日 (半年波段)": 120, "240日 (一年長線大底)": 240}
    sel_period = st.radio("🇹🇼 選擇大盤黃金分割計算週期：", list(period_map.keys()), horizontal=True)
    twii = get_twii_status(period_map[sel_period])

    if twii:
        is_bull = twii["close"] > twii["ma60"]
        market_mood = "🟢 季線之上（偏多操作，可正常低接）" if is_bull else "🚨 跌破季線（空頭修正，縮小試單）"
        if twii["rsi"] < 35 and twii["rsi"] > twii["prev_rsi"]:
            market_mood += " ｜ 🟢大盤超賣止跌勾頭"
        with st.expander(f"🇹🇼 台灣加權指數 (^TWII)：{twii['close']:,.0f} ({twii['diff']:+,.0f} / {twii['pct']:+.2f}%) ｜ {market_mood}", expanded=True):
            r1_1, r1_2, r1_3 = st.columns(3)
            r1_1.metric("大盤現價 / RSI", f"{twii['close']:,.0f}", f"14日 RSI: {twii['rsi']:.1f}")
            r1_2.metric("🛡️ 中期生命線(季線)", f"{twii['ma60']:,.0f}", f"距季線 {((twii['close']-twii['ma60'])/twii['ma60'])*100:+.1f}%")
            r1_3.metric("🏛️ 長線牛熊線(年線)", f"{twii['ma240']:,.0f}", f"距年線 {((twii['close']-twii['ma240'])/twii['ma240'])*100:+.1f}%")
            r2_1, r2_2, r2_3 = st.columns(3)
            r2_1.metric("📐 合理回檔(半山腰)", f"{twii['fib500']:,.0f}")
            r2_2.metric("📐 深度回檔(最後防線)", f"{twii['fib618']:,.0f}")
            r2_3.metric("⚠️ 恐慌破底測算線", f"{twii['fib_ext1272']:,.0f}")

render_live_market_panel()

if FINMIND_TOKEN:
    buy_top5, sell_top5, inst_date = get_top5_institutional(FINMIND_TOKEN)
    if not buy_top5.empty or not sell_top5.empty:
        with st.expander(f"🏦 三大法人合計買賣超 TOP 5（資料日期：{inst_date}）", expanded=False):
            c_buy, c_sell = st.columns(2)
            with c_buy:
                st.markdown("**🔥 買超前五名**")
                st.table(buy_top5.set_index("代號"))
            with c_sell:
                st.markdown("**❄️ 賣超前五名**")
                st.table(sell_top5.set_index("代號"))

tab1, tab2 = st.tabs(["🔔 持股黃金高賣與雙重防守", "📉 黃金分割＋季線共振低接掃描"])
with tab1:
    st.subheader("📋 持股黃金高賣與雙重防守即時監控（輸入代號與成本按 Enter 自動更新）")
    edited_df = st.data_editor(pd.DataFrame(DEFAULT_HOLDINGS), num_rows="dynamic", use_container_width=True, hide_index=True)

    @st.fragment(run_every=refresh_sec)
    def render_live_holdings():
        rows_out = []
        for _, r in edited_df.iterrows():
            code = str(r.get("代號", "")).strip()
            cost = float(r.get("買進成本", 0) or 0)
            if not code or cost <= 0:
                continue
            df, auto_name = fetch_single_stock(code)
            if df.empty or len(df) < 60:
                continue
            close = df["Close"]
            curr_c = float(close.iloc[-1])
            ma5, ma20, ma60 = float(close.rolling(5).mean().iloc[-1]), float(close.rolling(20).mean().iloc[-1]), float(close.rolling(60).mean().iloc[-1])
            rsi_s = calculate_rsi(close, 14)
            rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
            h20 = float(df["High"].iloc[-20:].max())
            h60, l60 = float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
            swing = h60 - l60
            fib_res_618, fib_ext_1272, fib_stop_786 = l60 + swing * 0.618, l60 + swing * 1.272, h60 - swing * 0.786
            dual_stop = max(ma60, fib_stop_786)
            stop_tag = "季線" if ma60 >= fib_stop_786 else "黃金0.786"
            pnl = ((curr_c - cost) / cost) * 100

            sigs = []
            if rsi < 30 and rsi > prev_rsi: sigs.append(f"🟢止跌反彈訊號(可試單/目標5日線{ma5:.2f})")
            elif rsi < 30: sigs.append("⏳還在跌先觀望(別急著接)")
            if rsi >= 75: sigs.append("🔥漲多過熱(建議分批高賣)")
            if curr_c >= fib_ext_1272 * 0.985 and pnl > 5: sigs.append("🚀達黃金延伸目標(建議高賣)")
            elif abs(curr_c - fib_res_618) / fib_res_618 <= 0.015 and pnl > 2: sigs.append("📐碰黃金壓力位(適合高賣)")
            elif curr_c >= h20 * 0.985 and pnl > 2: sigs.append("🎯漲回前波高點(適合高賣)")
            if pnl >= 3.0 and curr_c < ma5: sigs.append("📉跌破5日線(先鎖住獲利)")
            if curr_c < dual_stop or pnl <= -7.0: sigs.append(f"🚨跌破{stop_tag}{dual_stop:.2f}(建議停損)")

            rows_out.append({
                "代號": code, "名稱": auto_name, "成本": f"{cost:.2f}", "現價": f"{curr_c:.2f}", "損益": f"{pnl:+.2f}%",
                "RSI": f"{rsi:.1f}", "📐 黃金壓力(0.618)": f"{fib_res_618:.2f}", "🚀 黃金延伸(1.272)": f"{fib_ext_1272:.2f}",
                "🛡️ 雙重停損(季線/Fib0.786)": f"{dual_stop:.2f} ({stop_tag})", "白話操作建議": " / ".join(sigs) if sigs else "✅ 走勢正常，續抱觀察"
            })
        if rows_out:
            st.table(pd.DataFrame(rows_out).set_index("代號"))

    render_live_holdings()

with tab2:
    if not FINMIND_TOKEN:
        st.warning("⚠️ 請在左側邊欄輸入 FinMind Token 或設定 Streamlit Secrets。")
        st.stop()
    if st.button("🚀 開始全市場低接掃描", type="primary") or "market_cached" in st.session_state:
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
            ma20, ma60 = close.rolling(20).mean().iloc[-1], close.rolling(60).mean().iloc[-1]
            rsi = calculate_rsi(close, 14)
            curr_c, prev_c, curr_o, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(open_p.iloc[-1]), float(low_p.iloc[-1])
            curr_rsi, prev_rsi = float(rsi.iloc[-1]), float(rsi.iloc[-2])
            h20, h60, l60 = float(df["High"].iloc[-20:].max()), float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())

            if vol_ma5 < min_vol or curr_c < min_price or curr_c <= ma60 or ((h20 - curr_c) / h20) * 100 < min_pullback:
                continue
            fib_hit, fib_p, fib_dict, is_conf = check_fib_touch(curr_c, curr_l, h60, l60, ma60, fib_tol)
            if (req_fib and not fib_hit) or (req_conf_only and not is_conf) or (req_ma20 and curr_c >= ma20) or curr_rsi >= rsi_limit:
                continue
            if req_rebound and not (curr_rsi >= prev_rsi or curr_c > curr_o):
                continue
            swing = h60 - l60
            dual_stop = max(ma60, h60 - swing * 0.786)
            is_rsi_hook = (curr_rsi < 30) and (curr_rsi > prev_rsi)
            info = stock_map[t]
            passed.append({
                "ticker": t, "stock_id": info["id"], "name": info["name"], "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
                "rsi": curr_rsi, "fib_hit": fib_hit or "未觸及", "fib_price": fib_p, "fib_dict": fib_dict, "is_conf": is_conf,
                "is_rsi_hook": is_rsi_hook, "ma20": ma20, "dual_stop": dual_stop, "high_20d": h20,
                "up_ma20": ((ma20 - curr_c) / curr_c) * 100, "up_high": ((h20 - curr_c) / curr_c) * 100
            })

        passed = sorted(passed, key=lambda x: (not x["is_conf"], x["rsi"]))[:40]
        final_list = []
        with st.spinner(f"技術面 {len(passed)} 檔過關，核對三大法人買賣超中..."):
            for item in passed:
                net = check_inst_buy_5d(item["stock_id"], FINMIND_TOKEN)
                if req_inst and net <= 0:
                    continue
                item["net_5d"] = int(net)
                final_list.append(item)

        if not final_list:
            st.info("💡 目前條件無符合標的，請在左側放寬篩選條件。")
        else:
            df_res = pd.DataFrame(final_list).sort_values(by=["is_conf", "rsi", "net_5d"], ascending=[False, True, False]).head(top_n)
            def format_name(r):
                tags = []
                if r["is_conf"]: tags.append("🔥雙支撐")
                if r["is_rsi_hook"]: tags.append("🟢止跌反彈")
                return f"{r['name']} {' '.join(tags)}".strip()

            def format_buy_zone(r):
                low_b, high_b = r["fib_price"] * 0.99, r["fib_price"]
                status = "✅已到買區" if r["close"] <= high_b * 1.005 else "⏳等拉回"
                return f"{low_b:.1f}~{high_b:.1f} ({status})"

            disp = pd.DataFrame({
                "代號": df_res["stock_id"],
                "名稱": df_res.apply(format_name, axis=1),
                "現價": df_res["close"].map(lambda x: f"{x:.1f}"),
                "漲跌幅": df_res["pct"].map(lambda x: f"{x:+.2f}%"),
                "🟢 建議低接買點": df_res.apply(format_buy_zone, axis=1),
                "📐 黃金分割支撐": df_res.apply(lambda r: f"{r['fib_hit']} ({r['fib_price']:.1f})" if r['fib_hit'] != "未觸及" else "未觸及", axis=1),
                "14日 RSI": df_res["rsi"].map(lambda x: f"{x:.1f}"), "5日法人(張)": df_res["net_5d"].map(lambda x: f"{x:+,}"),
                "🟢 第一賣點(月線)": df_res.apply(lambda r: f"{r['ma20']:.1f} ({r['up_ma20']:+.1f}%)", axis=1),
                "🔴 第二賣點(前高)": df_res.apply(lambda r: f"{r['high_20d']:.1f} (+{r['up_high']:.1f}%)", axis=1),
                "🛡️ 雙重停損(季線/Fib0.786)": df_res["dual_stop"].map(lambda x: f"{x:.1f}"),
            })
            st.table(disp.set_index("代號"))

            opts = [f"{r['stock_id']} {r['name']}（現價 {r['close']:.1f} / 碰觸 {r['fib_hit']}）" for _, r in df_res.iterrows()]
            sel_row = df_res.iloc[opts.index(st.selectbox("選擇檢視低接標的走勢：", opts))]
            cdf = market_data[sel_row["ticker"]].copy().reset_index()
            date_col = cdf.columns.tolist().pop(0)
            fibs = sel_row["fib_dict"]
            cdf["20日月線"] = cdf["Close"].rolling(20).mean()
            cdf["60日季線"] = cdf["Close"].rolling(60).mean()
            cdf = cdf.tail(60)

            main_df = cdf[[date_col, "Close", "20日月線", "60日季線"]].rename(columns={date_col: "日期", "Close": "收盤價"})
            main_melt = main_df.melt("日期", var_name="指標", value_name="價格").dropna()
            main_chart = alt.Chart(main_melt).mark_line(strokeWidth=2.2).encode(
                x=alt.X("日期:T", title=""),
                y=alt.Y("價格:Q", scale=alt.Scale(zero=False), title="價格"),
                color=alt.Color("指標:N", scale=alt.Scale(
                    domain=["收盤價", "20日月線", "60日季線", "黃金 0.382", "黃金 0.500", "黃金 0.618"],
                    range=["#E60000", "#FF8C00", "#008000", "#6B4C35", "#0055FF", "#D0006F"]
                ))
            )

            fib_df = pd.DataFrame([
                {"指標": "黃金 0.382", "價格": fibs["0.382"]},
                {"指標": "黃金 0.500", "價格": fibs["0.500"]},
                {"指標": "黃金 0.618", "價格": fibs["0.618"]},
            ])
            fib_chart = alt.Chart(fib_df).mark_rule(strokeDash=list((6, 4)), strokeWidth=1.6).encode(
                y="價格:Q",
                color=alt.Color("指標:N")
            )

            st.altair_chart((main_chart + fib_chart).interactive(), use_container_width=True)
