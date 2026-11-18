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

def get_watch_pool():
    # 精選台股高流動性核心成分股與熱門電子/光通訊/傳產指標股（共 80 檔高效率掃描池）
    core_codes = [
        "2330.TW","2317.TW","2454.TW","2308.TW","2382.TW","3231.TW","6669.TW","2303.TW","3711.TW","2357.TW",
        "3008.TW","3019.TW","2340.TW","3450.TW","6442.TW","2379.TW","3034.TW","2327.TW","2376.TW","2301.TW",
        "2603.TW","2609.TW","2615.TW","2881.TW","2882.TW","2891.TW","2886.TW","2884.TW","2002.TW","1101.TW",
        "1513.TW","1519.TW","1503.TW","1504.TW","2356.TW","2353.TW","2324.TW","4938.TW","2383.TW","3037.TW",
        "3189.TW","8046.TW","2368.TW","6213.TW","3653.TW","3017.TW","3324.TWO","4908.TWO","3081.TWO","3363.TWO",
        "3163.TWO","4979.TWO","8069.TWO","8299.TWO","6488.TWO","5483.TWO","5347.TWO","3529.TWO","5274.TWO","6147.TWO"
    ]
    name_map = {}
    try:
        res = requests.get("https://githubusercontent.com", timeout=8)
        if res.status_code == 200:
            for c, info in res.json().items():
                n = info.get("name", "").strip() if isinstance(info, dict) else str(info).strip()
                if c and n: name_map[c] = f"{c} {n}"
    except Exception:
        pass
    return [(t, name_map.get(t.split(".")[0], t.split(".")[0])) for t in core_codes]

def scan_market(roll_win=35):
    pool = get_watch_pool()
    trends, dips = [], []
    for t, title in pool:
        try:
            df = yf.download(t, period="4mo", progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            curr = float(df["Close"].iat[-1])
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
                if dist_pct <= 0.08:
                    sl = min(f500, ma60) - 0.3 * atr
                    tp1, tp2 = hi, hi + diff * 0.272
                    avg_b = (b_hi + b_lo) / 2
                    rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 1.0
                    trends.append({"title": title, "price": curr, "peak": peak_d, "b_lo": b_lo, "b_hi": b_hi, "sl": sl, "tp1": tp1, "tp2": tp2, "tag": "前高/1.272延伸", "rsi": rsi, "rr": rr, "dist": dist_pct})
            else:
                anchor = f618
                b_hi, b_lo = anchor + min(0.15 * atr, anchor * 0.005), max(lo, anchor - min(0.25 * atr, anchor * 0.015))
                dist_pct = abs(curr - anchor) / anchor
                if dist_pct <= 0.10:
                    sl = lo - 0.5 * atr
                    tp1, tp2 = min(ma20, f500), f382
                    if tp1 > tp2: tp1, tp2 = tp2, tp1
                    avg_b = (b_hi + b_lo) / 2
                    rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 1.0
                    dips.append({"title": title, "price": curr, "peak": peak_d, "b_lo": b_lo, "b_hi": b_hi, "sl": sl, "tp1": tp1, "tp2": tp2, "tag": "0.5反彈/0.382減碼", "rsi": rsi, "rr": rr, "dist": dist_pct})
        except Exception:
            continue

    trends.sort(key=lambda x: (x["dist"], -x["rr"]))
    dips.sort(key=lambda x: (x["dist"], -x["rr"]))
    return trends[:3], dips[:3]

if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    trends, dips = scan_market()
    lines = [f"🚀 【AI 動態波浪黃金分割日報】\n日期：{today} ｜ 35日滾動防線\n" + "=" * 22]
    for label, group, start_i in [("🔥 【順勢主升段精選】（站穩季線）", trends, 1), ("🛡️ 【高點轉折低接精選】（0.618支撐）", dips, len(trends) + 1)]:
        if group:
            lines.append(label)
            for idx, s in enumerate(group, start_i):
                lines.append(f"{idx}. {s['title']}（現價 {s['price']:.1f}｜高點 {s['peak']}）\n   🟢 買進：{s['b_lo']:.1f}～{s['b_hi']:.1f}（批1:{s['b_hi']:.1f}/批2:{s['b_lo']:.1f}）\n   🔴 停損：{s['sl']:.1f} ｜ 🎯 賣點({s['tag']})：{s['tp1']:.1f} / {s['tp2']:.1f}\n   📊 賺賠比：{s['rr']:.1f}倍 ｜ RSI：{s['rsi']:.1f}")
    lines.append(f"📱 開啟動態防線圖表 APP：\n{APP_URL}")
    msg = "\n\n".join(lines)
    print(msg)
    send_line(msg)
