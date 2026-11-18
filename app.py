import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="AI 波浪黃金分割決策 APP", layout="centered")
st.title("📈 AI 波浪與黃金分割決策 APP")

BUILTIN = {
    "2330": ("2330.TW", "2330 台積電"), "2340": ("2340.TW", "2340 台亞"),
    "4908": ("4908.TWO", "4908 前鼎"), "3081": ("3081.TWO", "3081 聯亞"),
    "2317": ("2317.TW", "2317 鴻海"), "2454": ("2454.TW", "2454 聯發科"),
    "2382": ("2382.TW", "2382 廣達"), "3363": ("3363.TWO", "3363 上詮"),
    "6442": ("6442.TW", "6442 光聖"), "4979": ("4979.TWO", "4979 華星光")
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_maps():
    c_map, n_map = dict(BUILTIN), {v.split(" ")[1]: v for v in BUILTIN.values()}
    apis = [
        ("https://twse.com.tw", "Code", "Name", ".TW"),
        ("https://tpex.org.tw", "SecuritiesCompanyCode", "CompanyName", ".TWO")
    ]
    for url, k_c, k_n, sfx in apis:
        try:
            for item in requests.get(url, timeout=6).json():
                c, n = item.get(k_c, "").strip(), item.get(k_n, "").strip()
                if c and n:
                    c_map[c] = n_map[n] = (f"{c}{sfx}", f"{c} {n}")
        except Exception:
            pass
    return c_map, n_map

def find_stock(q, c_map, n_map):
    q = q.strip().upper()
    if q in c_map: return c_map[q]
    if q in n_map: return n_map[q]
    for name, info in n_map.items():
        if q in name: return info
    return (f"{q}.TW", q) if q.isdigit() else (None, q)

@st.cache_data(ttl=1800)
def fetch_data(ticker, raw_q):
    candidates = [ticker] if ticker else []
    if raw_q.isdigit(): candidates += [f"{raw_q}.TW", f"{raw_q}.TWO"]
    for t in dict.fromkeys(candidates):
        df = yf.download(t, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df, t
    return pd.DataFrame(), None

@st.cache_data(ttl=1800)
def fetch_inst(code):
    try:
        start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
        res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": code, "start_date": start}, timeout=8).json()
        if res.get("status") == 200 and res.get("data"):
            df = pd.DataFrame(res["data"])
            df["net"] = (df["buy"] - df["sell"]) / 1000.0
            piv = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
            out = pd.DataFrame(index=piv.index)
            out["外資"] = piv.get("Foreign_Investor", 0) + piv.get("Foreign_Dealer_Self", 0)
            out["投信"] = piv.get("Investment_Trust", 0)
            out["自營商"] = piv.get("Dealer_self", 0) + piv.get("Dealer_Hedging", 0)
            out["合計"] = out["外資"] + out["投信"] + out["自營商"]
            return out.tail(5)
    except Exception:
        pass
    return pd.DataFrame()

c_map, n_map = get_maps()
q = st.text_input("輸入台股代號或股名（例：台積電、2340、台亞、4908）", value="台積電")
days = st.slider("波段觀察天數", 30, 240, 90)
rr_min = st.number_input("最低要求賺賠比", 1.0, 5.0, 2.0, 0.5)

if st.button("🚀 啟動 AI 診斷", use_container_width=True):
    ticker, title = find_stock(q, c_map, n_map)
    df, f_ticker = fetch_data(ticker, q.strip())
    if df.empty:
        st.error(f"找不到「{q}」的資料，請確認輸入是否正確。")
    else:
        sub = df.tail(days)
        curr, hi, lo = float(sub['Close'].iloc[-1]), float(sub['High'].max()), float(sub['Low'].min())
        diff = hi - lo
        f382, f500, f618 = hi - diff * 0.382, hi - diff * 0.5, hi - diff * 0.618
        
        delta = df['Close'].diff()
        rsi = float((100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())).iloc[-1])
        tr = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        b_hi = max(f618, curr) if curr <= f618 * 1.03 else f618
        b_lo, sl = max(lo, f618 - 0.5 * atr), lo - 0.5 * atr
        tp1, tp2 = f382, (hi - diff * 0.236 if f500 <= curr else f500)
        avg_b = (b_hi + b_lo) / 2
        rr = (tp1 - avg_b) / (avg_b - sl) if avg_b > sl else 0
        code = f_ticker.split(".")[0]

        st.subheader(f"📊 {title} 交易計畫")
        c1, c2, c3 = st.columns(3)
        c1.metric("收盤價", f"{curr:.1f} 元")
        c2.metric("14日 RSI", f"{rsi:.1f}")
        c3.metric("預估賺賠比", f"{rr:.2f} 倍")

        b1, b2 = st.columns(2)
        b1.success(f"🟢 **建議買進區間**：{b_lo:.1f} ～ {b_hi:.1f} 元")
        b2.error(f"🔴 **嚴格停損價**：{sl:.1f} 元")
        t1, t2 = st.columns(2)
        t1.info(f"🎯 **第一停利 (0.382)**：{tp1:.1f} 元")
        t2.info(f"🏆 **第二停利 (0.500)**：{tp2:.1f} 元")

        if rr >= rr_min and rsi < 42 and curr <= f618 * 1.03:
            st.success("✅ **【強烈關注買進】** 回測 0.618 支撐且賺賠比達標！")
        else:
            st.info("ℹ️ **【建議觀望/等待回測】** 尚未同時滿足低檔支撐與賺賠比標準。")

        st.markdown("### 🏦 三大法人近 5 日買賣超（張）")
        df_i = fetch_inst(code)
        if not df_i.empty:
            st.bar_chart(df_i[["外資", "投信", "自營商"]], color=["#2980b9", "#e74c3c", "#f39c12"])
            st.dataframe(df_i.round(0).astype(int), use_container_width=True)
        else:
            st.caption("盤後籌碼更新中或暫無紀錄。")

        st.markdown("### 📉 走勢與黃金分割防線圖")
        chart = pd.DataFrame({"收盤價": sub['Close'], "第一停利": tp1, "黃金買進": f618, "停損線": sl})
        st.line_chart(chart, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
