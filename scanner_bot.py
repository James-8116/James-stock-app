import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

LINE_TOKEN = os.getenv("LINE_TOKEN", "")
LINE_USER_ID = os.getenv("LINE_USER_ID", "")
APP_URL = "https://streamlit.app"

def send_line(msg):
    if not LINE_TOKEN or not LINE_USER_ID:
        print("⚠️ 未設定 LINE 金鑰，略過推播")
        return
    url = "https://line.me"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": msg}]}
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    print("✅ LINE 推播成功" if res.status_code == 200 else f"❌ 推播失敗：{res.text}")

def get_active_stocks(min_lots=1000, min_price=15.0):
    stocks = []
    apis = [
        ("https://twse.com.tw", "Code", "Name", "TradeVolume", "ClosingPrice", ".TW"),
        ("https://tpex.org.tw", "SecuritiesCompanyCode", "CompanyName", "TradingShares", "Close", ".TWO")
    ]
    for url, k_c, k_n, k_v, k_p, sfx in apis:
        try:
            for item in requests.get(url, timeout=15).json():
                c, n = item.get(k_c, "").strip(), item.get(k_n, "").strip()
                if len(c) == 4 and c.isdigit() and not c.startswith("00"):
                    try:
                        lots = int(str(item.get(k_v, "0")).replace(",", "")) // 1000
                        price = float(str(item.get(k_p, "0")).replace(",", ""))
                        if lots >= min_lots and price >= min_price:
                            stocks.append({"ticker": f"{c}{sfx}", "title": f"{c} {n}"})
                    except ValueError:
                        continue
        except Exception:
            pass
    return stocks

def scan_market(days=90, rr_min=2.0):
    candidates = get_active_stocks()
    if not candidates:
        return []

    tickers = [c["ticker"] for c in candidates]
    name_map = {c["ticker"]: c["title"] for c in candidates}
    data = yf.download(tickers, period="1y", group_by="ticker", progress=False, threads=True)
    matched = []

    for t in tickers:
        try:
            df = data[t].dropna()
            if len(df) < days:
                continue
            sub = df.tail(days)
            curr = float(sub["Close"].iat[-1])
            hi, lo = float(sub["High"].max()), float(sub["Low"].min())
            diff = hi - lo
            if diff <= 0:
                continue

            f382, f500, f618 = hi - diff * 0.382, hi - diff * 0.500, hi - diff * 0.618

            # 篩選股價正處於 0.618 黃金買進支撐區附近（上下 2.5% 內）
            if not (f618 * 0.975 <= curr <= f618 * 1.025):
                continue

            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = float((100 - 100 / (1 + gain / loss)).iat[-1])
            if rsi >= 42:
                continue

            tr = np.max(pd.concat([
                df["High"] - df["Low"],
                np.abs(df["High"] - df["Close"].shift()),
                np.abs(df["Low"] - df["Close"].shift())
            ], axis=1), axis=1)
            atr = float(tr.rolling(14).mean().iat[-1])

            # 精準窄幅買進區間（總寬度嚴格限制在 2% 內）
            up_buf = min(0.15 * atr, f618 * 0.005)
            dn_buf = min(0.25 * atr, f618 * 0.015)
            b_hi = f618 + up_buf
            b_lo = max(lo, f618 - dn_buf)

            sl = lo - 0.5 * atr
            tp1 = f382
            tp2 = hi - diff * 0.236 if f500 <= curr else f500

            avg_b = (b_hi + b_lo) / 2
            rr = (tp1 - avg_b) / (avg_b - sl) if avg_b > sl else 0

            if rr >= rr_min:
                matched.append({
                    "title": name_map[t], "price": curr,
                    "b_lo": b_lo, "b_hi": b_hi, "sl": sl,
                    "tp1": tp1, "tp2": tp2, "rsi": rsi, "rr": rr
                })
        except Exception:
            continue

    matched.sort(key=lambda x: x["rr"], reverse=True)
    return matched[:8]

if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    results = scan_market()
    if not results:
        msg = (
            f"📊 【AI 量化選股日報】{today}\n\n"
            f"今日全市場掃描完畢，目前無同時符合「精準回測 0.618 支撐 + RSI<42 + 賺賠比≥2」之標的，建議保留現金觀望。\n\n"
            f"📱 手機看盤 APP：{APP_URL}"
        )
    else:
        lines = [
            f"🚀 【AI 波浪黃金分割精準買賣日報】\n"
            f"日期：{today}\n"
            f"策略：0.618窄幅支撐 | RSI<42 | 賺賠比≥2\n" + "=" * 22
        ]
        for idx, s in enumerate(results, 1):
            lines.append(
                f"{idx}. {s['title']}（現價 {s['price']:.1f} 元）\n"
                f"   🟢 精準買進區間：{s['b_lo']:.1f} ～ {s['b_hi']:.1f} 元\n"
                f"      （第一批：{s['b_hi']:.1f} ｜ 第二批：{s['b_lo']:.1f}）\n"
                f"   🔴 防洗盤停損價：{s['sl']:.1f} 元\n"
                f"   🎯 第一停利 (0.382)：{s['tp1']:.1f} 元\n"
                f"   🏆 第二停利 (0.500)：{s['tp2']:.1f} 元\n"
                f"   📊 賺賠比：{s['rr']:.1f} 倍 ｜ RSI：{s['rsi']:.1f}"
            )
        lines.append(f"📱 點擊開啟互動圖表 APP：\n{APP_URL}")
        msg = "\n\n".join(lines)

    print(msg)
    send_line(msg)
