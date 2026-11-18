import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

LINE_TOKEN = os.getenv("LINE_TOKEN", "").strip()
LINE_USER_ID = os.getenv("LINE_USER_ID", "").strip()
APP_URL = "https://" + "james-stockapp" + ".streamlit.app/"

def send_line(msg):
    if not LINE_TOKEN or not LINE_USER_ID: return
    push_url = "https://" + "api.line.me" + "/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    res = requests.post(push_url, headers=headers, json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": msg}]}, timeout=15)
    print("✅ LINE 推播成功！" if res.status_code == 200 else f"❌ 推播失敗：{res.text}")

def get_all_tw_stocks():
    pool = {}
    try:
        res = requests.get("https://finmindtrade.com", params={"dataset": "TaiwanStockInfo"}, timeout=12).json()
        if res.get("status") == 200:
            for it in res.get("data", []):
                c, n, m = str(it.get("stock_id", "")).strip(), str(it.get("stock_name", "")).strip(), str(it.get("type", "")).strip()
                if len(c) == 4 and c.isdigit() and not c.startswith("00") and n:
                    pool[f"{c}{'.TWO' if m == 'tpex' else '.TW'}"] = f"{c} {n}"
    except Exception:
        pass
    try:
        res = requests.get("https://githubusercontent.com", timeout=10)
        if res.status_code == 200:
            for c, info in res.json().items():
                n = info.get("name", "").strip() if isinstance(info, dict) else str(info).strip()
                m = info.get("type", "") if isinstance(info, dict) else ""
                if len(c) == 4 and c.isdigit() and not c.startswith("00") and n:
                    t = f"{c}{'.TWO' if '上櫃' in m or 'otc' in m.lower() else '.TW'}"
                    if t not in pool: pool[t] = f"{c} {n}"
    except Exception:
        pass
    return pool

def scan_market(roll_win=35):
    pool = get_all_tw_stocks()
    tickers = list(pool.keys())
    trends, dips = [], []

    for i in range(0, len(tickers), 250):
        batch = tickers[i:i+250]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
        except Exception:
            continue
        for t in batch:
            try:
                df = data[t].dropna() if len(batch) > 1 else data.dropna()
                if len(df) < 60: continue
                curr = float(df["Close"].iat[-1])
                if float(df["Volume"].tail(5).mean()) < 1000000 or curr < 15.0: continue

                sub = df.tail(roll_win)
                hi, lo = float(sub["High"].max()), float(sub["Low"].min())
                diff = hi - lo
                if diff <= 0: continue
                peak_d = sub["High"].idxmax().strftime("%m/%d")
                f382, f500, f618 = hi - diff * 0.382, hi - diff * 0.500, hi - diff * 0.618

                delta = df["Close"].diff()
                rsi = float((100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())).iat[-1])
                tr = np.max(pd.concat([df["High"] - df["Low"], np.abs(df["High"] - df["Close"].shift()), np.abs(df["Low"] - df["Close"].shift())], axis=1), axis=1)
                atr = float(tr.rolling(14).mean().iat[-1])
                ma20, ma60 = float(df["Close"].rolling(20).mean().iat[-1]), float(df["Close"].rolling(60).mean().iat[-1])

                if curr >= ma60 and ma20 >= ma60:
                    anchor = max(f382, ma20)
                    b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), anchor - min(0.25 * atr, anchor * 0.015)
                    dist_pct = abs(curr - anchor) / anchor
                    if dist_pct <= 0.06 and 38 <= rsi <= 72:
                        sl = min(f500, ma60) - 0.3 * atr
                        tp1, tp2 = hi, hi + diff * 0.272
                        avg_b = (b_hi + b_lo) / 2
                        rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 0.5
                        trends.append({"title": pool[t], "price": curr, "peak": peak_d, "b_lo": b_lo, "b_hi": b_hi, "sl": sl, "tp1": tp1, "tp2": tp2, "tag": "前高/1.272延伸", "rsi": rsi, "rr": rr, "dist": dist_pct})
                else:
                    anchor = f618
                    b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), max(lo, anchor - min(0.25 * atr, anchor * 0.015))
                    dist_pct = abs(curr - anchor) / anchor
                    if dist_pct <= 0.07 and rsi < 48:
                        sl = lo - 0.5 * atr
                        tp1, tp2 = min(ma20, f500), f382
                        if tp1 > tp2: tp1, tp2 = tp2, tp1
                        avg_b = (b_hi + b_lo) / 2
                        rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 0.5
                        dips.append({"title": pool[t], "price": curr, "peak": peak_d, "b_lo": b_lo, "b_hi": b_hi, "sl": sl, "tp1": tp1, "tp2": tp2, "tag": "0.5反彈/0.382減碼", "rsi": rsi, "rr": rr, "dist": dist_pct})
            except Exception:
                continue

    trends.sort(key=lambda x: (-x["rr"], x["dist"]))
    dips.sort(key=lambda x: (-x["rr"], x["dist"]))
    return trends[:3], dips[:3]

if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    trends, dips = scan_market()
    lines = [f"🚀 【AI 全市場動態波浪選股日報】\n日期：{today} ｜ 1,800檔上市櫃精選\n" + "=" * 22]
    for label, group, start_i in [("🔥 【順勢主升段 Top 3】（站穩季線）", trends, 1), ("🛡️ 【高點轉折低接 Top 3】（0.618支撐）", dips, len(trends) + 1)]:
        if group:
            lines.append(label)
            for idx, s in enumerate(group, start_i):
                lines.append(f"{idx}. {s['title']}（現價 {s['price']:.1f}｜高點 {s['peak']}）\n   🟢 買進：{s['b_lo']:.1f}～{s['b_hi']:.1f}（批1:{s['b_hi']:.1f}/批2:{s['b_lo']:.1f}）\n   🔴 停損：{s['sl']:.1f} ｜ 🎯 賣點({s['tag']})：{s['tp1']:.1f} / {s['tp2']:.1f}\n   📊 賺賠比：{s['rr']:.1f}倍 ｜ RSI：{s['rsi']:.1f}")
    lines.append(f"📱 開啟動態防線圖表 APP：\n{APP_URL}")
    msg = "\n\n".join(lines)
    print(msg)
    send_line(msg)
