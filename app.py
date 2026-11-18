import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="AI 波浪黃金分割決策 APP", layout="centered")
st.title("📈 AI 波浪與黃金分割決策 APP")
st.caption("整合：波段高低點 + 費波那契回檔 + RSI動能 + 量能確認 + 三大法人主力動態")

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_mapping():
    c_map, n_map = {}, {}
    try:
        for item in requests.get("https://twse.com.tw", timeout=8).json():
            c, n = item.get("Code", "").strip(), item.get("Name", "").strip()
            if c and n:
                c_map[c] = (f"{c}.TW", f"{c} {n}", c)
                n_map[n] = (f"{c}.TW", f"{c} {n}", c)
    except Exception:
        pass
    try:
        for item in requests.get("https://tpex.org.tw", timeout=8).json():
            c, n = item.get("SecuritiesCompanyCode", "").strip(), item.get("CompanyName", "").strip()
            if c and n:
                c_map[c] = (f"{c}.TWO", f"{c} {n}", c)
                n_map[n] = (f"{c}.TWO", f"{c} {n}", c)
    except Exception:
        pass
    if len(c_map) < 500:
        try:
            res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInfo"}, timeout=10).json()
            if res.get("status") == 200:
                for item in res.get("data", []):
                    c, n, m = str(item.get("stock_id", "")).strip(), str(item.get("stock_name", "")).strip(), str(item.get("type", "")).strip()
                    if len(c) == 4 and c.isdigit() and n:
                        sfx = ".TWO" if m == "tpex" else ".TW"
                        c_map[c] = (f"{c}{sfx}", f"{c} {n}", c)
                        n_map[n] = (f"{c}{sfx}", f"{c} {n}", c)
        except Exception:
            pass
    return c_map, n_map

def get_mapping():
    c_map, n_map = _fetch_mapping()
    if not c_map:
        _fetch_mapping.clear()
    return c_map, n_map

def resolve_query(q, c_map, n_map):
    q = q.strip().upper()
    if not q:
        return None, None, None
    if q in c_map:
        return c_map[q]
    if q in n_map:
        return n_map[q]
    matches = [info for name, info in n_map.items() if q in name]
    if matches:
        matches.sort(key=lambda x: len(x[1]))
        return matches[0]
    return None, q, q

@st.cache_data(ttl=1800)
def get_price_df(ticker, fallback):
    if ticker:
        df = yf.download(ticker, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, ticker
    if fallback.isdigit():
        for sfx in [".TW", ".TWO"]:
            df = yf.download(f"{fallback}{sfx}", period="1y", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, f"{fallback}{sfx}"
    return pd.DataFrame(), None

@st.cache_data(ttl=1800)
def get_inst_df(code):
    try:
        start_d = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
        res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": code, "start_date": start_d}, timeout=10).json()
        if res.get("status") == 200 and res.get("data"):
            df_i = pd.DataFrame(res["data"])
            df_i["net"] = (df_i["buy"] - df_i["sell"]) / 1000.0
            piv = df_i.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
            out = pd.DataFrame(index=piv.index)
            out["外資"] = piv.get("Foreign_Investor", 0) + piv.get("Foreign_Dealer_Self", 0)
            out["投信"] = piv.get("Investment_Trust", 0)
            out["自營商"] = piv.get("Dealer_self", 0) + piv.get("Dealer_Hedging", 0)
            out["三大法人合計"] = out["外資"] + out["投信"] + out["自營商"]
            return out.tail(5)
    except Exception:
        pass
    return pd.DataFrame()

def calc_metrics(df, days):
    sub = df.tail(days).copy()
    curr = float(sub['Close'].iloc[-1])
    hi, lo = float(sub['High'].max()), float(sub['Low'].min())
    diff = hi - lo
    f382, f500, f618 = hi - diff * 0.382, hi - diff * 0.500, hi - diff * 0.618
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = float((100 - (100 / (1 + gain / loss))).iloc[-1])
    
    tr = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    bias20 = ((curr - ma20) / ma20) * 100
    v_ma5 = float(df['Volume'].rolling(5).mean().iloc[-1])
    v_ratio = float(df['Volume'].iloc[-1]) / v_ma5 if v_ma5 > 0 else 1.0
    
    b_hi = max(f618, curr) if curr <= f618 * 1.03 else f618
    b_lo = max(lo, f618 - 0.5 * atr)
    sl = lo - 0.5 * atr
    tp1 = f382
    tp2 = hi - diff * 0.236 if f500 <= curr else f500
    avg_b = (b_hi + b_lo) / 2
    rr = (tp1 - avg_b) / (avg_b - sl) if (avg_b - sl) > 0 else 0
    return {"price": curr, "high": hi, "low": lo, "f382": f382, "f618": f618, "rsi": rsi, "bias20": bias20, "v_ratio": v_ratio, "b_lo": b_lo, "b_hi": b_hi, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr}

c_map, n_map = get_mapping()
u_query = st.text_input("輸入台股代號或股名（例如：2340、台亞、前鼎、聯亞）", value="2340")
days = st.slider("波段觀察天數（決定高低點基準）", 30, 240, 90)
rr_min = st.number_input("最低要求風險報酬比（賺賠比）", 1.0, 5.0, 2.0, 0.5)

if st.button("🚀 啟動 AI 量化與籌碼診斷", use_container_width=True):
    with st.spinner("正在運算波浪結構、黃金分割與三大法人主力籌碼..."):
        r_ticker, d_title, p_code = resolve_query(u_query, c_map, n_map)
        df, f_ticker = get_price_df(r_ticker, u_query.strip())
        if df.empty:
            st.error(f"找不到「{u_query}」的相關資料，請確認輸入的代號或中文股名是否正確。")
        else:
            res = calc_metrics(df, days)
            t_text = d_title if r_ticker else f_ticker
            c_code = p_code if p_code else str(f_ticker).split(".")[0]
            st.subheader(f"📊 {t_text} 量化交易計畫")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最新收盤價", f"{res['price']:.1f} 元", f"月線乖離 {res['bias20']:.1f}%")
            c2.metric("14日 RSI", f"{res['rsi']:.1f}", "低檔轉折區" if res['rsi'] < 42 else "中性/偏強")
            c3.metric("今日量能比", f"{res['v_ratio']:.2f} 倍", "爆量換手" if res['v_ratio'] >= 1.5 else "量縮整理")
            
            st.divider()
            st.markdown("### 🎯 具體買賣區間與停損停利點")
            b1, b2 = st.columns(2)
            b1.success(f"🟢 **建議分批買進區間**\n\n**{res['b_lo']:.1f} ～ {res['b_hi']:.1f} 元**")
            b2.error(f"🔴 **防洗盤嚴格停損價**\n\n**{res['sl']:.1f} 元**（低點 - 0.5 ATR）")
            t1, t2 = st.columns(2)
            t1.info(f"🎯 **第一停利位 (0.382)**\n\n**{res['tp1']:.1f} 元**（建議減碼一半）")
            t2.info(f"🏆 **第二停利位 (0.500)**\n\n**{res['tp2']:.1f} 元**（波段全數出場）")
            
            st.write(f"**波段基準區間：** 最高 `{res['high']:.1f} 元` ｜ 最低 `{res['low']:.1f} 元` ｜ **預估賺賠比：** `{res['rr']:.2f} 倍`")
            if res['rr'] >= rr_min and res['rsi'] < 42 and res['price'] <= res['f618'] * 1.03:
                st.success("✅ **【強烈關注買進】** 股價已回測 0.618 黃金支撐區，RSI 處於低檔且賺賠比大於設定門檻！")
            elif res['rr'] < rr_min:
                st.warning(f"⚠️ **【建議觀望】** 目前距離停損點過遠或上方反彈空間不足，賺賠比（{res['rr']:.2f}）未達 {rr_min} 倍標準。")
            else:
                st.info("ℹ️ **【等待回測】** 股價尚未落入 0.618 黃金買進區或 RSI 尚未超賣，建議耐心等待拉回買點。")
            
            st.divider()
            st.markdown("### 🏦 三大法人主力進出動態（近 5 交易日，單位：張）")
            df_i = get_inst_df(c_code)
            if not df_i.empty:
                n5, f5, tr5 = df_i["三大法人合計"].sum(), df_i["外資"].sum(), df_i["投信"].sum()
                i1, i2, i3 = st.columns(3)
                i1.metric("近5日法人合計", f"{n5:+.0f} 張", "主力偏多吸籌" if n5 > 0 else "主力偏空調節")
                i2.metric("近5日外資動向", f"{f5:+.0f} 張")
                i3.metric("近5日投信動向", f"{tr5:+.0f} 張")
                st.bar_chart(df_i[["外資", "投信", "自營商"]], color=["#2980b9", "#e74c3c", "#f39c12"])
                st.dataframe(df_i.round(0).astype(int), use_container_width=True)
            else:
                st.info("盤後法人籌碼更新中（通常於交易日下午 15:30～16:30 陸續公布），或暫無近期法人進出紀錄。")
            
            y_sfx = "TWO" if ".TWO" in str(f_ticker) else "TW"
            st.markdown(f"👉 [點擊查看 {c_code} 券商分點主力買賣超明細（Yahoo股市）](https://yahoo.com{c_code}.{y_sfx}/broker-trading)")
            
            st.divider()
            st.markdown("### 📉 走勢與黃金分割關鍵防線圖")
            c_df = pd.DataFrame(index=df.tail(days).index)
            c_df['收盤價'] = df['Close'].tail(days)
            c_df['第一停利 (0.382)'] = res['tp1']
            c_df['黃金買進線 (0.618)'] = res['f618']
            c_df['嚴格停損線'] = res['sl']
            st.line_chart(c_df, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
