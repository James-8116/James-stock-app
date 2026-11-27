import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="台股 AI 波段監控與精準突破選股", page_icon="🚀", layout="wide")

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
FINMIND_URL = "https://api." + "://finmindtrade.com"
TWSE_T86_URL = "https://openapi." + "twse.com.tw/v1/fund/T86"
TW_TZ = timezone(timedelta(hours=8))

TECH_INDUSTRIES = {
    "半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
    "通信網路業", "電子通路業", "資訊服務業", "其他電子業"
}

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


def calculate_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def run_v6_backtest(df):
    """V6 精準突破 2 年回測引擎 (含隔日跳空滑價與 0.5% 稅費修正)"""
    if len(df) < 130:
        return 0, 0.0, 0.0

    close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    vol_ma5 = vol.rolling(5).mean()
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(df, 14)

    trades = []
    in_pos, entry_p, stop_p, highest_p = False, 0.0, 0.0, 0.0
    fee = 0.005

    for i in range(122, len(df) - 1):
        c, o, l, h = float(close.iloc[i]), float(open_p.iloc[i]), float(low.iloc[i]), float(high.iloc[i])
        c_prev = float(close.iloc[i - 1])
        v, v_ma5 = float(vol.iloc[i]), float(vol_ma5.iloc[i])
        m20, m60, m120 = float(ma20.iloc[i]), float(ma60.iloc[i]), float(ma120.iloc[i])
        m60_20d_ago = float(ma60.iloc[i - 20])
        r = float(rsi.iloc[i])
        curr_atr = float(atr.iloc[i])

        if pd.isna(curr_atr) or m60_20d_ago <= 0:
            continue

        prev_20d_high = float(close.iloc[i - 20 : i].max())
        box_h = float(high.iloc[i - 5 : i].max())
        box_l = float(low.iloc[i - 5 : i].min())
        box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0

        if not in_pos:
            is_strong_trend = (c > m20 > m60 > m120) and ((m60 - m60_20d_ago) / m60_20d_ago >= 0.025)
            is_tight_box = box_range <= 0.085
            body = c - o
            upper_shadow = h - c
            pct_chg = (c - c_prev) / c_prev
            is_clean_breakout = (
                (c > prev_20d_high)
                and (body > 0)
                and (upper_shadow <= body * 0.35)
                and (0.02 <= pct_chg <= 0.08)
                and (v > v_ma5 * 1.25)
                and (55 <= r <= 73)
            )
            if is_strong_trend and is_tight_box and is_clean_breakout:
                in_pos = True
                entry_p = c
                highest_p = c
                stop_p = max(l * 0.99, c - 1.8 * curr_atr, c * 0.945)
        else:
            highest_p = max(highest_p, h)
            trailing_stop = max(stop_p, highest_p - 2.2 * curr_atr)
            if l <= trailing_stop:
                exit_p = min(o, trailing_stop)
                trades.append(((exit_p - entry_p) / entry_p - fee) * 100)
                in_pos = False
            elif c < m20:
                trades.append(((c - entry_p) / entry_p - fee) * 100)
                in_pos = False

    if not trades:
        return 0, 0.0, 0.0
    wins = [t for t in trades if t > 0]
    return len(trades), len(wins) / len(trades) * 100, sum(trades) / len(trades)
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
def get_tw_stock_list(token, tech_only=True):
    try:
        res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo", "token": token}, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        latest_date = pd.to_datetime(df["date"]).max()
        cutoff = (latest_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        cond = (
            df["type"].isin(["twse", "tpex"])
            & (df["stock_id"].str.len() == 4)
            & (df["stock_id"].str.isdigit())
            & (df["date"] >= cutoff)
        )
        if tech_only and "industry_category" in df.columns:
            cond = cond & (df["industry_category"].isin(TECH_INDUSTRIES))
        df = df[cond].drop_duplicates(subset=["stock_id"], keep="last")
        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {
                "id": r["stock_id"],
                "name": r["stock_name"],
                "industry": r.get("industry_category", "電子科技"),
            }
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_top5_institutional(token, tech_only=True):
    stock_map = get_tw_stock_list(token, tech_only=tech_only)
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
    batch_size = 200
    bar = st.progress(0, text="下載 2 年日 K 線（供即時突破掃描與 V6 歷史回測）...")
    total = (len(tickers) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="2y", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t in data.columns.get_level_values(0):
                    df = data[t].dropna(subset=["Close", "Volume"])
                    if len(df) >= 125:
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
    st.header("🚀 V6 精準突破參數微調")
    only_tech = st.toggle("💻 鎖定電子 / AI / 半導體族群 (與LINE同步)", value=True)
    rsi_min, rsi_max = st.slider("14日 RSI 強勢攻擊區間", 45, 80, (55, 73), 1)
    max_box_pct = st.slider("突破前 5 日箱型最大震幅 (%)", 4.0, 15.0, 8.5, 0.5) / 100
    min_ma60_slope = st.slider("季線近 20 日最低上揚幅度 (%)", 0.0, 6.0, 2.5, 0.5) / 100
    min_vol_ratio = st.slider("突破當日最低爆量倍數 (vs 5日均量)", 1.0, 2.5, 1.25, 0.05)
    min_vol = st.number_input("5日均量下限 (張)", 300, 10000, 1000, 100)
    min_price = st.number_input("最低股價 (元)", 10, 200, 20, 5)
    st.divider()
    req_clean_k = st.checkbox("🛡️ 防隔日沖濾網 (上影線 <= 實體35%)", value=True)
    req_inst = st.checkbox("🏦 強制近 5 日三大法人買超 > 0", value=False)
    top_n = st.slider("顯示精選檔數 (TOP N)", 3, 15, 10)
    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")
st.title("🚀 台股 AI 波段監控與 V6 精準突破選股看板")
tab1, tab2 = st.tabs(["🔔 持股與觀察股階梯支撐監控", "🚀 V6 電子/AI 箱型蓄勢帶量突破 TOP 10"])

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
        market_mood = "🟢 季線之上（多頭順風，啟動 V6 強勢突破策略）" if is_bull else "🚨 跌破季線（空頭修正，嚴禁追高突破）"
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
    buy_top5, sell_top5, inst_date = get_top5_institutional(FINMIND_TOKEN, tech_only=only_tech)
    scope_label = "電子/半導體" if only_tech else "全市場"
    if not buy_top5.empty or not sell_top5.empty:
        with st.expander(f"🏦 {scope_label}三大法人合計買賣超 TOP 5（資料日期：{inst_date}）", expanded=False):
            c_buy, c_sell = st.columns(2)
            with c_buy:
                st.markdown("**🔥 買超前五名**")
                st.table(buy_top5.set_index("代號"))
            with c_sell:
                st.markdown("**❄️ 賣超前五名**")
                st.table(sell_top5.set_index("代號"))

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
                "代號": code, "名稱": auto_name,
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
with tab2:
    if not FINMIND_TOKEN:
        st.warning("⚠️ 請在左側邊欄輸入 FinMind Token 或設定 Streamlit Secrets。")
        st.stop()
    b1, b2 = st.columns(2)
    run_scan = b1.button("🚀 開始 V6 電子/AI 精準突破 TOP 10 掃描", type="primary", use_container_width=True)
    force_refresh = b2.button("🔄 強制重抓最新報價", use_container_width=True)
    if force_refresh:
        download_all_market_data.clear()
        st.session_state["market_cached"] = True

    if run_scan or force_refresh or "market_cached" in st.session_state:
        stock_map = get_tw_stock_list(FINMIND_TOKEN, tech_only=only_tech)
        if not stock_map:
            st.error("無法取得台股清單，請檢查 Token。")
            st.stop()
        market_data = download_all_market_data(list(stock_map.keys()))
        st.session_state["market_cached"] = True

        passed = []
        for t, df in market_data.items():
            if t not in stock_map:
                continue
            close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
            vol_ma5_s = (vol / 1000).rolling(5).mean()
            vol_ma5 = float(vol_ma5_s.iloc[-1])
            curr_v = float(vol.iloc[-1]) / 1000
            curr_c = float(close.iloc[-1])

            if vol_ma5 < min_vol or curr_c < min_price:
                continue

            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60_s = close.rolling(60).mean()
            ma60, ma60_20d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-21])
            ma120 = float(close.rolling(120).mean().iloc[-1])
            curr_rsi = float(calculate_rsi(close, 14).iloc[-1])
            curr_atr = float(calculate_atr(df, 14).iloc[-1])

            if pd.isna(curr_atr) or ma60_20d_ago <= 0:
                continue

            c_prev = float(close.iloc[-2])
            curr_o, curr_h, curr_l = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
            prev_20d_high = float(close.iloc[-21:-1].max())
            box_h = float(high.iloc[-6:-1].max())
            box_l = float(low.iloc[-6:-1].min())
            box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0

            # 1. 四線多頭排列 + 季線強斜率
            if not ((curr_c > ma20 > ma60 > ma120) and ((ma60 - ma60_20d_ago) / ma60_20d_ago >= min_ma60_slope)):
                continue
            # 2. 箱型壓縮蓄勢
            if box_range > max_box_pct:
                continue
            # 3. 突破20日收盤新高 + 出量 + RSI強勢區
            body = curr_c - curr_o
            upper_shadow = curr_h - curr_c
            pct_chg = (curr_c - c_prev) / c_prev
            if not (curr_c > prev_20d_high and body > 0 and (0.02 <= pct_chg <= 0.08) and curr_v > vol_ma5 * min_vol_ratio and (rsi_min <= curr_rsi <= rsi_max)):
                continue
            if req_clean_k and upper_shadow > body * 0.35:
                continue

            init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
            trail_ref = max(ma20, curr_c - 2.2 * curr_atr)
            bt_cnt, bt_win, bt_ret = run_v6_backtest(df)
            info = stock_map[t]

            passed.append({
                "ticker": t, "stock_id": info["id"], "name": info["name"],
                "industry": info["industry"].replace("業", "").replace("設備", ""),
                "close": curr_c, "pct": pct_chg * 100, "rsi": curr_rsi,
                "break_lvl": prev_20d_high, "init_stop": init_stop, "trail_ref": trail_ref,
                "box_pct": box_range * 100, "vol_ratio": curr_v / vol_ma5 if vol_ma5 > 0 else 1.0,
                "bt_cnt": bt_cnt, "bt_win": bt_win, "bt_ret": bt_ret,
            })

        passed = sorted(passed, key=lambda x: (-x["bt_ret"], -x["vol_ratio"]))[:25]
        final_list = []
        with st.spinner(f"V6 技術突破條件 {len(passed)} 檔過關，核對三大法人數據並計算綜合評分中..."):
            for item in passed:
                net = check_inst_buy_5d(item["stock_id"], FINMIND_TOKEN)
                if req_inst and net <= 0:
                    continue
                item["net_5d"] = int(net)
                score = 0
                if item["net_5d"] > 0:
                    score += min(35, item["net_5d"] / 80)
                if item["bt_cnt"] >= 1 and item["bt_ret"] > 0:
                    score += min(35, 15 + item["bt_ret"] * 2)
                score += min(30, item["vol_ratio"] * 12)
                item["score"] = round(score, 1)
                final_list.append(item)

        if not final_list:
            st.info("💡 今日無符合 V6「箱型蓄勢＋實體飽滿帶量突破」之標的（寧缺勿濫），可在左側適度放寬開關。")
        else:
            df_res = pd.DataFrame(final_list).sort_values(by=["score"], ascending=False).head(top_n)
            format_name = lambda r: f"{r['name']} ({r['industry']}) 🚀{r['vol_ratio']:.1f}x量"
            format_bt = lambda r: f"{r['bt_win']:.0f}% ({r['bt_cnt']}次 / 均{r['bt_ret']:+.1f}%)" if r["bt_cnt"] > 0 else "2年首現突破"

            disp = pd.DataFrame({
                "代號": df_res["stock_id"],
                "名稱/族群": df_res.apply(format_name, axis=1),
                "現價": df_res["close"].map(lambda x: f"{x:.1f}"),
                "漲跌幅": df_res["pct"].map(lambda x: f"{x:+.2f}%"),
                "📊 V6回測(勝率/均報)": df_res.apply(format_bt, axis=1),
                "5日法人(張)": df_res["net_5d"].map(lambda x: f"{x:+,}"),
                "🎯 20日突破頸線": df_res["break_lvl"].map(lambda x: f"{x:.1f}"),
                "🛡️ 初始硬停損": df_res["init_stop"].map(lambda x: f"{x:.1f}"),
                "📈 ATR移動停利線": df_res["trail_ref"].map(lambda x: f"{x:.1f}"),
                "14日 RSI": df_res["rsi"].map(lambda x: f"{x:.1f}"),
            })
            st.table(disp.set_index("代號"))

            opts = [f"#{i+1} {r['stock_id']} {r['name']}（現價 {r['close']:.1f} / 均報 {r['bt_ret']:+.1f}%）" for i, (_, r) in enumerate(df_res.iterrows())]
            sel_row = df_res.iloc[opts.index(st.selectbox("選擇檢視 V6 突破標的走勢：", opts))]
            cdf = market_data[sel_row["ticker"]].copy().reset_index()
            date_col = cdf.columns[0]
            cdf["20日月線"] = cdf["Close"].rolling(20).mean()
            cdf["60日季線"] = cdf["Close"].rolling(60).mean()
            cdf = cdf.tail(60)

            main_melt = cdf[[date_col, "Close", "20日月線", "60日季線"]].rename(columns={date_col: "日期", "Close": "收盤價"}).melt("日期", var_name="指標", value_name="價格").dropna()
            main_chart = alt.Chart(main_melt).mark_line(strokeWidth=2.2).encode(
                x=alt.X("日期:T", title=""),
                y=alt.Y("價格:Q", scale=alt.Scale(zero=False), title="價格"),
                color=alt.Color("指標:N", scale=alt.Scale(
                    domain=["收盤價", "20日月線", "60日季線", "20日突破頸線", "初始硬停損", "ATR移動停利"],
                    range=["#E60000", "#FF8C00", "#008000", "#0055FF", "#15803D", "#9333EA"],
                )),
            )
            lvl_df = pd.DataFrame([
                {"指標": "20日突破頸線", "價格": sel_row["break_lvl"]},
                {"指標": "初始硬停損", "價格": sel_row["init_stop"]},
                {"指標": "ATR移動停利", "價格": sel_row["trail_ref"]},
            ])
            lvl_chart = alt.Chart(lvl_df).mark_rule(strokeDash=[5, 5], strokeWidth=1.6).encode(
                y="價格:Q",
                color=alt.Color("指標:N")
            )
            st.altair_chart((main_chart + lvl_chart).interactive(), use_container_width=True)
