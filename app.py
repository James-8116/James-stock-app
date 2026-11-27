import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="台股 AI 雙策略波段監控看板", page_icon="📊", layout="wide")

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

FINMIND_TOKEN = "".join(str(st.secrets.get("FINMIND_TOKEN", os.environ.get("FINMIND_TOKEN", ""))).split())
FINMIND_URL = "https://api." + "://finmindtrade.com"
TW_TZ = timezone(timedelta(hours=8))

TECH_INDUSTRIES = {
    "半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
    "通信網路業", "電子通路業", "資訊服務業", "其他電子業"
}

DEFAULT_HOLDINGS = [
    {"代號": "2330", "買進成本": 0.0},
    {"代號": "2303", "買進成本": 0.0},
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
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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
        rsi = float(calculate_rsi(close, 14).iloc[-1])
        sub = df.iloc[-lookback_days:]
        h_val, l_val = float(sub["High"].max()), float(sub["Low"].min())
        swing = h_val - l_val
        return {
            "close": curr_c, "diff": curr_c - prev_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
            "ma60": ma60, "rsi": rsi,
            "fib500": h_val - swing * 0.500, "fib618": h_val - swing * 0.618, "fib786": h_val - swing * 0.786,
        }
    except Exception:
        return None


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
                "id": r["stock_id"], "name": r["stock_name"], "industry": r.get("industry_category", "電子科技")
            }
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def download_market_data(tickers):
    all_dfs = {}
    batch_size = 200
    bar = st.progress(0, text="下載 1 年日 K 線中...")
    total = (len(tickers) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="1y", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t in data.columns.get_level_values(0):
                    df = data[t].dropna(subset=["Close", "Volume"])
                    if len(df) >= 125:
                        all_dfs[t] = df
        except Exception:
            pass
        bar.progress((idx + 1) / total)
        time.sleep(0.15)
    bar.empty()
    return all_dfs


def check_inst_buy_5d(stock_id, token):
    if not token:
        return 0
    start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start, "token": token}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=8).json()
        if res.get("msg") == "success" and res.get("data"):
            df = pd.DataFrame(res["data"])
            df["net"] = df["buy"] - df["sell"]
            recent = sorted(df["date"].unique())[-5:]
            return int(round(df[df["date"].isin(recent)]["net"].sum() / 1000))
    except Exception:
        pass
    return 0
with st.sidebar:
    st.header("🎯 雙引擎選股設定")
    only_tech = st.toggle("💻 鎖定電子 / AI / 半導體族群", value=True)
    min_vol = st.number_input("5日均量下限 (張)", 300, 10000, 1000, 100)
    min_price = st.number_input("最低股價 (元)", 10, 200, 20, 5)
    req_inst = st.checkbox("🏦 強制近 5 日三大法人買超 > 0", value=False)
    top_each = st.slider("每組顯示檔數 (TOP N)", 3, 10, 5)
    st.divider()
    if not FINMIND_TOKEN:
        FINMIND_TOKEN = st.text_input("輸入 FinMind Token", type="password")

st.title("📊 台股 AI 雙策略選股與監控看板")

twii = get_twii_status(60)
if twii:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🇹🇼 加權指數", f"{twii['close']:,.0f}", f"{twii['diff']:+,.0f} ({twii['pct']:+.2f}%)")
    c2.metric("🛡️ 季線 (60MA)", f"{twii['ma60']:,.0f}", "多頭順風" if twii["close"] > twii["ma60"] else "空頭防守")
    c3.metric("📐 大盤 Fib 0.500", f"{twii['fib500']:,.0f}")
    c4.metric("📐 大盤 Fib 0.618", f"{twii['fib618']:,.0f}")

tab1, tab2 = st.tabs(["🚀 雙引擎選股 TOP 10 (V6突破 ＋ 黃金分割共振)", "🔔 自選股階梯支撐監控"])

with tab1:
    if not FINMIND_TOKEN:
        st.warning("⚠️ 請在左側輸入 FinMind Token。")
        st.stop()

    b1, b2 = st.columns(2)
    run_scan = b1.button("🚀 執行雙引擎 TOP 10 掃描", type="primary", use_container_width=True)
    force_refresh = b2.button("🔄 強制更新最新報價", use_container_width=True)
    if force_refresh:
        download_market_data.clear()

    if run_scan or force_refresh or "dual_cached" in st.session_state:
        stock_map = get_tw_stock_list(FINMIND_TOKEN, tech_only=only_tech)
        market_data = download_market_data(list(stock_map.keys()))
        st.session_state["dual_cached"] = True

        breakouts, rebounds = [], []
        for t, df in market_data.items():
            if t not in stock_map:
                continue
            close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
            vol_ma5 = float((vol / 1000).rolling(5).mean().iloc[-1])
            curr_v = float(vol.iloc[-1]) / 1000
            curr_c = float(close.iloc[-1])
            if vol_ma5 < min_vol or curr_c < min_price:
                continue

            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60_s = close.rolling(60).mean()
            ma60, ma60_20d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-21])
            ma120 = float(close.rolling(120).mean().iloc[-1])
            rsi_s = calculate_rsi(close, 14)
            rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
            curr_atr = float(calculate_atr(df, 14).iloc[-1])
            if pd.isna(curr_atr) or ma60_20d_ago <= 0:
                continue

            c_prev = float(close.iloc[-2])
            curr_o, curr_h, curr_l = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
            pct_chg = (curr_c - c_prev) / c_prev * 100
            vol_ratio = curr_v / vol_ma5 if vol_ma5 > 0 else 1.0
            info = stock_map[t]

            # 引擎 A：V6 突破
            prev_20d_high = float(close.iloc[-21:-1].max())
            box_h, box_l = float(high.iloc[-6:-1].max()), float(low.iloc[-6:-1].min())
            box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0
            if (
                (curr_c > ma20 > ma60 > ma120) and ((ma60 - ma60_20d_ago) / ma60_20d_ago >= 0.025)
                and (box_range <= 0.085) and (curr_c > prev_20d_high)
                and (curr_c > curr_o) and ((curr_h - curr_c) <= (curr_c - curr_o) * 0.35)
                and (2.0 <= pct_chg <= 8.0) and (vol_ratio >= 1.25) and (55 <= rsi <= 73)
            ):
                init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
                trail_ref = max(ma20, curr_c - 2.2 * curr_atr)
                breakouts.append({
                    "代號": info["id"], "名稱": f"{info['name']} 🚀突破",
                    "現價": f"{curr_c:.1f} ({pct_chg:+.1f}%)",
                    "建議買點": f"{curr_c*0.995:.1f}~{curr_c*1.01:.1f}",
                    "賣點目標": f"{trail_ref:.1f} / {curr_c*1.09:.1f}",
                    "雙重防守": f"{init_stop:.1f}", "sort_key": vol_ratio
                })

            # 引擎 B：黃金分割＋季線共振低接
            h60, l60 = float(high.iloc[-60:].max()), float(low.iloc[-60:].min())
            swing = h60 - l60
            if swing > 0:
                fib500, fib618, fib786 = h60 - swing * 0.500, h60 - swing * 0.618, h60 - swing * 0.786
                hit_fib = fib618 if (abs(curr_c - fib618)/fib618 <= 0.025 or abs(curr_l - fib618)/fib618 <= 0.025) else (
                          fib500 if (abs(curr_c - fib500)/fib500 <= 0.025 or abs(curr_l - fib500)/fib500 <= 0.025) else None)
                if hit_fib and (rsi <= 44) and (rsi >= prev_rsi - 1.0 or (min(curr_c, curr_o) - curr_l) > abs(curr_c - curr_o) * 0.7):
                    is_conf = abs(hit_fib - ma60) / ma60 <= 0.025
                    dual_stop = max(fib786, curr_c * 0.945)
                    rebounds.append({
                        "代號": info["id"], "名稱": f"{info['name']} {'🔥共振' if is_conf else '📐低接'}",
                        "現價": f"{curr_c:.1f} ({pct_chg:+.1f}%)",
                        "建議買點": f"{hit_fib*0.99:.1f}~{hit_fib*1.005:.1f}",
                        "賣點目標": f"{ma20:.1f} / {h60:.1f}",
                        "雙重防守": f"{dual_stop:.1f}", "sort_key": (100 if is_conf else 0) + (50 - rsi)
                    })

        breakouts = sorted(breakouts, key=lambda x: x["sort_key"], reverse=True)[:10]
        rebounds = sorted(rebounds, key=lambda x: x["sort_key"], reverse=True)[:10]

        with st.spinner("核對三大法人買賣超中..."):
            for grp in (breakouts, rebounds):
                for item in grp:
                    net = check_inst_buy_5d(item["代號"], FINMIND_TOKEN)
                    item["5日法人"] = f"{net:+,}張"
                    item["_net"] = net

        if req_inst:
            breakouts = [x for x in breakouts if x["_net"] > 0]
            rebounds = [x for x in rebounds if x["_net"] > 0]

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"🚀 V6 強勢突破 TOP {top_each}")
            if breakouts:
                df_b = pd.DataFrame(breakouts[:top_each])[["代號", "名稱", "5日法人", "現價", "建議買點", "賣點目標", "雙重防守"]]
                st.table(df_b.set_index("代號"))
            else:
                st.info("今日無符合 V6 嚴格突破條件之標的。")

        with col_right:
            st.subheader(f"📉 黃金分割＋季線共振低接 TOP {top_each}")
            if rebounds:
                df_r = pd.DataFrame(rebounds[:top_each])[["代號", "名稱", "5日法人", "現價", "建議買點", "賣點目標", "雙重防守"]]
                st.table(df_r.set_index("代號"))
            else:
                st.info("今日無符合黃金分割止跌共振之標的。")

with tab2:
    st.subheader("📋 自選股階梯支撐與雙重防守速查")
    edited_df = st.data_editor(pd.DataFrame(DEFAULT_HOLDINGS), num_rows="dynamic", use_container_width=True, hide_index=True)
    rows_out = []
    for _, r in edited_df.iterrows():
        code = str(r.get("代號", "")).strip()
        if not code:
            continue
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df.empty:
            df = yf.download(f"{code}.TWO", period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            continue
        curr_c = float(df["Close"].iloc[-1])
        ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
        ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
        h60, l60 = float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
        swing = h60 - l60
        rows_out.append({
            "代號": code, "現價": f"{curr_c:.1f}", "月線": f"{ma20:.1f}", "季線": f"{ma60:.1f}",
            "Fib 0.500": f"{h60 - swing*0.5:.1f}", "Fib 0.618": f"{h60 - swing*0.618:.1f}", "防守 0.786": f"{h60 - swing*0.786:.1f}"
        })
    if rows_out:
        st.table(pd.DataFrame(rows_out).set_index("代號"))
