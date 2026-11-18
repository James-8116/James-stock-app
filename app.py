import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import altair as alt

st.set_page_config(page_title="AI 動態波浪決策 APP", layout="centered")
st.title("📈 AI 波浪與黃金分割決策 APP")

BUILTIN = {
    "2330": ("2330.TW", "2330 台積電"), "2340": ("2340.TW", "2340 台亞"),
    "4908": ("4908.TWO", "4908 前鼎"), "3081": ("3081.TWO", "3081 聯亞"),
    "3008": ("3008.TW", "3008 大立光"), "3019": ("3019.TW", "3019 亞光"),
    "2317": ("2317.TW", "2317 鴻海"), "2454": ("2454.TW", "2454 聯發科"),
    "8069": ("8069.TWO", "8069 元太"), "3324": ("3324.TWO", "3324 雙鴻")
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_maps():
    c_map, n_map = dict(BUILTIN), {v.split(" "): v for v in BUILTIN.values()}
    try:
        res = requests.get("https://githubusercontent.com", timeout=8)
        if res.status_code == 200:
            for c, info in res.json().items():
                n = info.get("name", "").strip() if isinstance(info, dict) else str(info).strip()
                m = info.get("type", "") if isinstance(info, dict) else ""
                if len(c) == 4 and c.isdigit() and n:
                    c_map[c] = n_map[n] = (f"{c}{'.TWO' if '上櫃' in m or 'otc' in m.lower() else '.TW'}", f"{c} {n}")
    except Exception:
        pass
    try:
        res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInfo"}, timeout=8).json()
        if res.get("status") == 200:
            for it in res.get("data", []):
                c, n, m = str(it.get("stock_id", "")).strip(), str(it.get("stock_name", "")).strip(), str(it.get("type", "")).strip()
                if len(c) == 4 and c.isdigit() and n:
                    c_map[c] = n_map[n] = (f"{c}{'.TWO' if m == 'tpex' else '.TW'}", f"{c} {n}")
    except Exception:
        pass
    return c_map, n_map

def find_stock(q, c_map, n_map):
    q = q.strip().upper()
    if q in c_map: return c_map[q]
    if q in n_map: return n_map[q]
    for name, info in sorted(n_map.items(), key=lambda x: len(x)):
        if q in name: return info
    return (f"{q}.TW", q) if q.isdigit() else (None, q)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker, raw_q):
    cands = ([ticker] if ticker else []) + ([f"{raw_q}.TW", f"{raw_q}.TWO"] if raw_q.isdigit() else [])
    for t in dict.fromkeys(cands):
        df = yf.download(t, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns.name = None
            return df, t
    return pd.DataFrame(), None

@st.cache_data(ttl=1800, show_spinner=False)
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
q = st.text_input("輸入台股代號或股名（例：亞光、大立光、元太、雙鴻、聯亞）", value="亞光")

# 即時預覽：輸入完股名立刻秀出最新收盤價與漲跌
ticker, title = find_stock(q, c_map, n_map)
df, f_ticker = fetch_data(ticker, q.strip())
if not df.empty and len(df) >= 2:
    p_now, p_prev = float(df["Close"].iat[-1]), float(df["Close"].iat[-2])
    chg = p_now - p_prev
    st.info(f"📌 **{title}** ｜ 最新收盤價：**`{p_now:.1f} 元`**（{chg:+.1f} 元 / {chg/p_prev*100:+.2f}%）")
elif q.strip():
    st.warning(f"⚠️ 暫時找不到「{q}」的報價，請確認代號或股名。")

col_in1, col_in2 = st.columns(2)
roll_win = col_in1.slider("動態滾動波浪週期", 20, 90, 35)
custom_buy = col_in2.number_input("自訂試算買進價（0 = 用系統低接價）", min_value=0.0, value=0.0, step=0.5)

if st.button("🚀 啟動 AI 動態波浪診斷", use_container_width=True):
    if df.empty:
        st.error(f"找不到「{q}」的資料，請確認輸入是否正確。")
    else:
        df = df.copy()
        df['roll_hi'], df['roll_lo'] = df['High'].rolling(roll_win).max(), df['Low'].rolling(roll_win).min()
        diff = df['roll_hi'] - df['roll_lo']
        df['dyn_382'], df['dyn_500'], df['dyn_618'] = df['roll_hi'] - diff * 0.382, df['roll_hi'] - diff * 0.500, df['roll_hi'] - diff * 0.618
        df['ma20'] = df['Close'].rolling(20).mean()
        delta = df['Close'].diff()
        df['rsi'] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())
        tr = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['動態停利壓力'] = np.where(df['Close'] >= df['ma20'], df['roll_hi'], df['dyn_382'])
        df['動態買進支撐'] = np.where(df['Close'] >= df['ma20'], df['dyn_382'], df['dyn_618'])
        df['動態移動停損'] = np.where(df['Close'] >= df['ma20'], df['dyn_500'] - 0.3 * df['atr'], df['roll_lo'] - 0.5 * df['atr'])

        latest = df.iloc[-1]
        curr, hi, lo = float(latest['Close']), float(latest['roll_hi']), float(latest['roll_lo'])
        peak_date = df.tail(roll_win)['High'].idxmax().strftime("%Y-%m-%d")
        f382, f500, f618 = float(latest['dyn_382']), float(latest['dyn_500']), float(latest['dyn_618'])
        ma20, rsi, atr = float(latest['ma20']), float(latest['rsi']), float(latest['atr'])

        if curr >= ma20 and curr >= f382:
            status, anchor = "🔥 【主升段移動停利模式】", max(f382, ma20)
            b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), anchor - min(0.25 * atr, anchor * 0.015)
            sl, tp1, tp2, tp1_l, tp2_l = float(latest['動態移動停損']), hi, hi + (hi - lo) * 0.272, "波段前高", "1.272延伸"
        elif curr < ma20 and curr >= f618:
            status, anchor = f"⚠️ 【高點 ({peak_date}) 後轉折修正】", f618
            b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), max(lo, anchor - min(0.25 * atr, anchor * 0.015))
            sl, tp1, tp2, tp1_l, tp2_l = lo - 0.5 * atr, min(ma20, f500), f382, "月線/0.5反彈", "0.382減碼"
        else:
            status = f"🚨 【高點 ({peak_date}) 後跌破 0.618 支撐】"
            b_hi, b_lo, sl = lo + min(0.2 * atr, lo * 0.01), lo, lo - 0.5 * atr
            tp1, tp2, tp1_l, tp2_l = f618, f500, "0.618逃命點", "0.500壓力點"

        calc_entry = custom_buy if custom_buy > 0 else (b_hi + b_lo) / 2
        risk_amt, reward_amt = calc_entry - sl, tp1 - calc_entry
        rr = reward_amt / risk_amt if risk_amt > 0 else 0
        code = f_ticker.split(".")

        st.subheader(f"📊 {title} 動態交易計畫")
        st.info(f"**{status}**（近 {roll_win} 日高點：`{peak_date}` 的 `{hi:.1f} 元`）")
        c1, c2, c3 = st.columns(3)
        c1.metric("最新收盤價", f"{curr:.1f} 元", f"月線 {ma20:.1f}")
        c2.metric("14日 RSI", f"{rsi:.1f}")
        c3.metric(f"動態賺賠比（基準 {calc_entry:.1f}）", f"{rr:.2f} 倍", f"獲利 {reward_amt/calc_entry*100:+.1f}% / 風險 {-risk_amt/calc_entry*100:.1f}%")

        b1, b2 = st.columns(2)
        b1.success(f"🟢 **低接買進帶**：{b_lo:.1f} ～ {b_hi:.1f} 元\n\n（第一批：{b_hi:.1f} ｜ 第二批：{b_lo:.1f}）")
        b2.error(f"🔴 **移動停損價**：{sl:.1f} 元")
        t1, t2 = st.columns(2)
        t1.warning(f"🎯 **第一賣點 ({tp1_l})**：{tp1:.1f} 元")
        t2.warning(f"🏆 **第二賣點 ({tp2_l})**：{tp2:.1f} 元")

        st.markdown("### 🏦 三大法人近 5 日買賣超（張）")
        df_i = fetch_inst(code)
        if not df_i.empty:
            st.bar_chart(df_i[["外資", "投信", "自營商"]], color=["#2980b9", "#e74c3c", "#f39c12"])
            st.dataframe(df_i.round(0).astype(int), use_container_width=True)
        else:
            st.caption("盤後籌碼更新中或暫無紀錄。")

        st.markdown("### 📉 動態滾動買賣防線走勢圖（近 90 日）")
        sub90 = df.tail(90)
        dates = pd.to_datetime(sub90.index)
        l_map = {"收盤價": sub90["Close"].to_numpy(float), "動態停利壓力": sub90["動態停利壓力"].to_numpy(float), "動態買進支撐": sub90["動態買進支撐"].to_numpy(float), "動態移動停損": sub90["動態移動停損"].to_numpy(float)}
        cdf = pd.concat([pd.DataFrame({"日期": dates, "指標線": k, "價格": v}) for k, v in l_map.items()], ignore_index=True).dropna()
        chart = alt.Chart(cdf).mark_line(strokeWidth=2).encode(
            x=alt.X("日期:T", title="日期"), y=alt.Y("價格:Q", scale=alt.Scale(domain=[float(cdf["價格"].min()) * 0.96, float(cdf["價格"].max()) * 1.04]), title="價格"),
            color=alt.Color("指標線:N", scale=alt.Scale(domain=list(l_map.keys()), range=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])),
            tooltip=["日期:T", "指標線:N", alt.Tooltip("價格:Q", format=".1f")]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
