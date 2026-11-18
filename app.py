import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import altair as alt

st.set_page_config(page_title="AI 動態波浪黃金分割決策 APP", layout="centered")
st.title("📈 AI 波浪與黃金分割決策 APP")
st.caption("升級版：全市場上市櫃中文股名搜尋 + 逐日滾動重算高低點 + 移動停利防線")

BUILTIN = {
    "2330": ("2330.TW", "2330 台積電"), "2340": ("2340.TW", "2340 台亞"),
    "4908": ("4908.TWO", "4908 前鼎"), "3081": ("3081.TWO", "3081 聯亞"),
    "3008": ("3008.TW", "3008 大立光"), "2317": ("2317.TW", "2317 鴻海"),
    "2454": ("2454.TW", "2454 聯發科"), "3363": ("3363.TWO", "3363 上詮"),
    "6442": ("6442.TW", "6442 光聖"), "4979": ("4979.TWO", "4979 華星光")
}

@st.cache_data(ttl=86400, show_spinner=False)
def _load_full_maps():
    c_map, n_map = dict(BUILTIN), {}
    for v in BUILTIN.values():
        _, title = v
        _, name = title.split(" ")
        n_map[name] = v
    try:
        res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInfo"}, timeout=10).json()
        if res.get("status") == 200:
            for item in res.get("data", []):
                c, n, m = str(item.get("stock_id", "")).strip(), str(item.get("stock_name", "")).strip(), str(item.get("type", "")).strip()
                if len(c) == 4 and c.isdigit() and n:
                    sfx = ".TWO" if m == "tpex" else ".TW"
                    c_map[c] = n_map[n] = (f"{c}{sfx}", f"{c} {n}")
    except Exception:
        pass
    for url, kc, kn, sfx in [
        ("https://twse.com.tw", "Code", "Name", ".TW"),
        ("https://tpex.org.tw", "SecuritiesCompanyCode", "CompanyName", ".TWO")
    ]:
        try:
            for item in requests.get(url, timeout=6).json():
                c, n = item.get(kc, "").strip(), item.get(kn, "").strip()
                if c and n:
                    c_map[c] = n_map[n] = (f"{c}{sfx}", f"{c} {n}")
        except Exception:
            pass
    return c_map, n_map

def get_maps():
    c_map, n_map = _load_full_maps()
    if len(c_map) < 100:
        _load_full_maps.clear()
    return c_map, n_map

def find_stock(q, c_map, n_map):
    q = q.strip().upper()
    if q in c_map: return c_map[q]
    if q in n_map: return n_map[q]
    matches = [(name, info) for name, info in n_map.items() if q in name]
    if matches:
        matches.sort(key=lambda x: len(x))
        return matches
    if not q.isdigit():
        try:
            y_res = requests.get("https://yahoo.com", params={"q": q, "quotesCount": 5}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).json()
            for item in y_res.get("quotes", []):
                sym = item.get("symbol", "")
                if sym.endswith(".TW") or sym.endswith(".TWO"):
                    return (sym, f"{sym.split('.')} {item.get('shortname', q)}")
        except Exception:
            pass
    return (f"{q}.TW", q) if q.isdigit() else (None, q)

@st.cache_data(ttl=1800)
def fetch_data(ticker, raw_q):
    candidates = [ticker] if ticker else []
    if raw_q.isdigit(): candidates += [f"{raw_q}.TW", f"{raw_q}.TWO"]
    for t in dict.fromkeys(candidates):
        df = yf.download(t, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns.name = None
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
q = st.text_input("輸入台股代號或股名（例：大立光、元太、雙鴻、聯亞、2330）", value="大立光")
roll_win = st.slider("動態滾動波浪週期（天數越短越貼近近期轉折）", 20, 90, 35)
rr_min = st.number_input("最低要求賺賠比", 1.0, 5.0, 2.0, 0.5)

if st.button("🚀 啟動 AI 動態波浪診斷", use_container_width=True):
    ticker, title = find_stock(q, c_map, n_map)
    df, f_ticker = fetch_data(ticker, q.strip())
    if df.empty:
        st.error(f"找不到「{q}」的資料，請確認輸入是否正確。")
    else:
        df = df.copy()
        df['roll_hi'] = df['High'].rolling(roll_win).max()
        df['roll_lo'] = df['Low'].rolling(roll_win).min()
        df['roll_diff'] = df['roll_hi'] - df['roll_lo']
        df['dyn_382'] = df['roll_hi'] - df['roll_diff'] * 0.382
        df['dyn_500'] = df['roll_hi'] - df['roll_diff'] * 0.500
        df['dyn_618'] = df['roll_hi'] - df['roll_diff'] * 0.618
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma60'] = df['Close'].rolling(60).mean()

        delta = df['Close'].diff()
        df['rsi'] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())
        tr = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
        df['atr'] = tr.rolling(14).mean()

        df['動態停利壓力'] = np.where(df['Close'] >= df['ma20'], df['roll_hi'], df['dyn_382'])
        df['動態買進支撐'] = np.where(df['Close'] >= df['ma20'], df['dyn_382'], df['dyn_618'])
        df['動態移動停損'] = np.where(df['Close'] >= df['ma20'], df['dyn_500'] - 0.3 * df['atr'], df['roll_lo'] - 0.5 * df['atr'])

        latest = df.iloc[-1]
        curr = float(latest['Close'])
        hi, lo = float(latest['roll_hi']), float(latest['roll_lo'])
        peak_date = df.tail(roll_win)['High'].idxmax().strftime("%Y-%m-%d")
        f382, f500, f618 = float(latest['dyn_382']), float(latest['dyn_500']), float(latest['dyn_618'])
        ma20, rsi, atr = float(latest['ma20']), float(latest['rsi']), float(latest['atr'])

        if curr >= ma20 and curr >= f382:
            status = "🔥 【主升段移動停利模式】（沿月線與 0.382 強勢推進）"
            anchor = max(f382, ma20)
            b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), anchor - min(0.25 * atr, anchor * 0.015)
            sl = float(latest['動態移動停損'])
            tp1, tp2, tp1_lbl, tp2_lbl = hi, hi + (hi - lo) * 0.272, "波段前高", "1.272 延伸目標"
        elif curr < ma20 and curr >= f618:
            status = f"⚠️ 【高點 ({peak_date}) 後轉折修正】（已跌破月線，下修賣點為反彈減碼位）"
            anchor = f618
            b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), max(lo, anchor - min(0.25 * atr, anchor * 0.015))
            sl = lo - 0.5 * atr
            tp1, tp2, tp1_lbl, tp2_lbl = min(ma20, f500), f382, "月線/0.500反彈賣點", "0.382強勢減碼點"
        else:
            status = f"🚨 【高點 ({peak_date}) 後跌破 0.618 支撐】（空方修正波，反彈皆看賣點）"
            b_hi, b_lo, sl = lo + min(0.2 * atr, lo * 0.01), lo, lo - 0.5 * atr
            tp1, tp2, tp1_lbl, tp2_lbl = f618, f500, "0.618反彈逃命點", "0.500減碼壓力點"

        avg_b = (b_hi + b_lo) / 2
        rr = (tp1 - avg_b) / (avg_b - sl) if avg_b > sl else 0
        code, _ = f_ticker.split(".")

        st.subheader(f"📊 {title} 動態交易計畫")
        st.info(f"**最新狀態判斷：** {status}\n\n（近 {roll_win} 日波段高點出現於 **`{peak_date}`** 的 **`{hi:.1f} 元`**，防線已隨最新行情自動修正）")

        c1, c2, c3 = st.columns(3)
        c1.metric("最新收盤價", f"{curr:.1f} 元", f"月線 {ma20:.1f}")
        c2.metric("14日 RSI", f"{rsi:.1f}")
        c3.metric("動態賺賠比", f"{rr:.2f} 倍")

        b1, b2 = st.columns(2)
        b1.success(f"🟢 **動態低接買進帶**：{b_lo:.1f} ～ {b_hi:.1f} 元\n\n（第一批：{b_hi:.1f} ｜ 第二批：{b_lo:.1f}）")
        b2.error(f"🔴 **動態移動停損價**：{sl:.1f} 元")
        t1, t2 = st.columns(2)
        t1.warning(f"🎯 **修正第一賣點 ({tp1_lbl})**：{tp1:.1f} 元")
        t2.warning(f"🏆 **修正第二賣點 ({tp2_lbl})**：{tp2:.1f} 元")

        st.markdown("### 🏦 三大法人近 5 日買賣超（張）")
        df_i = fetch_inst(code)
        if not df_i.empty:
            st.bar_chart(df_i[["外資", "投信", "自營商"]], color=["#2980b9", "#e74c3c", "#f39c12"])
            st.dataframe(df_i.round(0).astype(int), use_container_width=True)
        else:
            st.caption("盤後籌碼更新中或暫無紀錄。")

        st.markdown("### 📉 動態滾動買賣防線走勢圖（近 90 交易日）")
        sub90 = df.tail(90)
        dates = pd.to_datetime(sub90.index)
        line_map = {
            "收盤價": sub90["Close"].to_numpy(dtype=float),
            "動態停利壓力": sub90["動態停利壓力"].to_numpy(dtype=float),
            "動態買進支撐": sub90["動態買進支撐"].to_numpy(dtype=float),
            "動態移動停損": sub90["動態移動停損"].to_numpy(dtype=float),
        }
        chart_df = pd.concat([pd.DataFrame({"日期": dates, "指標線": lbl, "價格": vals}) for lbl, vals in line_map.items()], ignore_index=True).dropna()
        y_min, y_max = float(chart_df["價格"].min()) * 0.96, float(chart_df["價格"].max()) * 1.04

        chart = alt.Chart(chart_df).mark_line(strokeWidth=2).encode(
            x=alt.X("日期:T", title="日期"),
            y=alt.Y("價格:Q", scale=alt.Scale(domain=[y_min, y_max]), title="價格 (元)"),
            color=alt.Color("指標線:N", scale=alt.Scale(domain=["收盤價", "動態停利壓力", "動態買進支撐", "動態移動停損"], range=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])),
            tooltip=[alt.Tooltip("日期:T", title="日期"), "指標線:N", alt.Tooltip("價格:Q", format=".1f")]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
