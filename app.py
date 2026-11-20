import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
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

components.html(
    """
    <script>
    const doc = window.parent.document;
    const setNumeric = () => {
        doc.querySelectorAll('input[type="text"], input[type="search"]').forEach(el => {
            el.setAttribute('inputmode', 'numeric');
            el.setAttribute('pattern', '[0-9.]*');
        });
    };
    setNumeric();
    new MutationObserver(setNumeric).observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
    height=0,
)

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
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
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
        sub = df.iloc[-lookback_days:] if len(df) >= lookback_days else df
        h_val, l_val = float(sub["High"].max()), float(sub["Low"].min())
        swing = h_val - l_val
        return {
            "close": curr_c, "diff": curr_c - prev_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
            "ma60": ma60, "ma240": ma240, "rsi": rsi, "prev_rsi": prev_rsi,
            "fib382": h_val - swing * 0.382,
            "fib500": h_val - swing * 0.500,
            "fib618": h_val - swing * 0.618,
            "fib_stop786": max(0.0, h_val - swing * 0.786),
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
        latest_date = pd.to_datetime(df["date"]).max()
        cutoff = (latest_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        df = df[
            df["type"].isin(["twse", "tpex"])
            & (df["stock_id"].str.len() == 4)
            & (df["stock_id"].str.isdigit())
            & (df["date"] >= cutoff)
        ]
        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {"id": r["stock_id"], "name": r["stock_name"]}
            for _, r in df.iterrows()
        }
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
                    if len(df) >= 66:
                        all_dfs[t] = df
        except Exception:
            pass
        bar.progress((idx + 1) / total, text=f"下載進度：{idx + 1} / {total}")
        time.sleep(0.2)
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
    df = yf.download(ticker_str, period="18mo", progress=False)
    if df.empty:
        ticker_str = code + ".TWO"
        df = yf.download(ticker_str, period="18mo", progress=False)
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
    req_ma60_up = st.checkbox("強制季線斜率向上 (避免頭部下彎)", value=True)
    req_rebound = st.checkbox("必須當日止跌 (收紅K或RSI勾頭)", value=True)
    max_stop_pct = st.slider("最大停損空間上限 (%)", 3.0, 12.0, 6.5, 0.5) / 100
    req_inst = st.checkbox("強制近 5 日三大法人買超 > 0", value=True)
    top_n = st.slider("顯示精選檔數", 3, 15, 5)
    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")

st.title("📊 台股 AI 波段監控與選股看板")

period_map = {"20日 (月波段)": 20, "60日 (季波段)": 60, "120日 (半年波段)": 120, "240日 (年波段)": 240}
sel_twii_period = st.radio("🇹🇼 選擇【大盤】黃金分割計算週期：", list(period_map.keys()), index=1, horizontal=True, key="twii_period_radio")
twii_days = period_map[sel_twii_period]


@st.fragment(run_every=refresh_sec)
def render_live_market_panel(days):
    now_tw = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    status_txt = "🟢 即時自動更新中 (每60秒)" if auto_refresh else "⏸️ 已暫停自動更新"
    st.caption(f"🕒 最後報價更新：{now_tw}（台灣時間）｜ {status_txt}")
    twii = get_twii_status(days)

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
            r2_1.metric("📐 合理回檔(0.500)", f"{twii['fib500']:,.0f}")
            r2_2.metric("📐 深度回檔(0.618)", f"{twii['fib618']:,.0f}")
            r2_3.metric("🧱 極限防守線(0.786)", f"{twii['fib_stop786']:,.0f}")


render_live_market_panel(twii_days)

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

tab1, tab2 = st.tabs(["🔔 持股與觀察股階梯支撐監控", "📉 黃金分割＋季線共振低接掃描"])
with tab1:
    sel_stock_period = st.radio("📈 選擇【個股】黃金分割計算週期：", list(period_map.keys()), index=1, horizontal=True, key="stock_period_radio")
    stock_days = period_map[sel_stock_period]
    st.subheader(f"📋 個股階梯支撐與雙重防守監控（目前基準：{sel_stock_period}，輸入代號即可查詢，成本選填）")
    edited_df = st.data_editor(pd.DataFrame(DEFAULT_HOLDINGS), num_rows="dynamic", use_container_width=True, hide_index=True)

    @st.fragment(run_every=refresh_sec)
    def render_live_holdings(days):
        rows_out = []
        for _, r in edited_df.iterrows():
            code = str(r.get("代號", "")).strip()
            if not code or code.lower() == "none":
                continue
            raw_cost = r.get("買進成本")
            has_cost = pd.notna(raw_cost) and float(raw_cost or 0) > 0
            cost = float(raw_cost) if has_cost else 0.0

            df, auto_name = fetch_single_stock(code)
            if df.empty or len(df) < 60:
                continue
            close = df["Close"]
            curr_c = float(close.iloc[-1])
            ma5, ma60 = float(close.rolling(5).mean().iloc[-1]), float(close.rolling(60).mean().iloc[-1])
            rsi_s = calculate_rsi(close, 14)
            rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
            h20 = float(df["High"].iloc[-20:].max())
            sub = df.iloc[-days:] if len(df) >= days else df
            h_val, l_val = float(sub["High"].max()), float(sub["Low"].min())
            swing = h_val - l_val

            fib_500 = h_val - swing * 0.500
            fib_618 = h_val - swing * 0.618
            fib_stop_786 = max(0.0, h_val - swing * 0.786)
            dual_stop = max(ma60, fib_stop_786)
            stop_tag = "季線" if ma60 >= fib_stop_786 else "Fib0.786"
            pnl = ((curr_c - cost) / cost) * 100 if has_cost else None

            sigs = []
            if rsi < 30 and rsi > prev_rsi:
                sigs.append(f"🟢止跌勾頭(目標5日線{ma5:.1f})")
            elif rsi < 30:
                sigs.append("⏳超賣修正中(別急著接)")
            if rsi >= 75:
                sigs.append("🔥漲多過熱(建議分批高賣)")

            if abs(curr_c - fib_500) / fib_500 <= 0.02:
                sigs.append("📐回測半山腰(0.500)")
            elif abs(curr_c - fib_618) / fib_618 <= 0.02:
                sigs.append("📐觸及深度支撐(0.618)")
            elif curr_c < fib_stop_786:
                sigs.append(f"🚨跌破0.786防線(下看波段低點{l_val:.1f})")
            elif curr_c < fib_618:
                sigs.append(f"⚠️破0.618(下看0.786防線{fib_stop_786:.1f})")

            if has_cost:
                if curr_c >= h20 * 0.985 and pnl > 2:
                    sigs.append("🎯漲回前高(適合高賣)")
                if pnl >= 3.0 and curr_c < ma5:
                    sigs.append("📉破5日線(鎖住獲利)")
                if curr_c < dual_stop or pnl <= -7.0:
                    sigs.append(f"🚨跌破{stop_tag}{dual_stop:.1f}(建議停損)")
            else:
                if curr_c < ma60:
                    sigs.append(f"🛡️空頭排列(季線壓力{ma60:.1f})")

            rows_out.append({
                "代號": code,
                "名稱": auto_name,
                "現價": f"{curr_c:.2f}",
                "📐 合理回檔(0.500)": f"{fib_500:.2f}",
                "📐 深度回檔(0.618)": f"{fib_618:.2f}",
                "🧱 極限防守線(0.786)": f"{fib_stop_786:.2f}",
                "🛡️ 雙重防守線": f"{dual_stop:.2f} ({stop_tag})",
                "白話操作建議": " / ".join(sigs) if sigs else "✅ 走勢正常，續抱觀察",
                "成本": f"{cost:.2f}" if has_cost else "-",
                "損益": f"{pnl:+.2f}%" if has_cost else "-",
                "RSI": f"{rsi:.1f}",
            })
        if rows_out:
            st.table(pd.DataFrame(rows_out).set_index("代號"))

    render_live_holdings(stock_days)
