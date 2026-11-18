import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

LINE_TOKEN = os.getenv("LINE_TOKEN", "").strip()
LINE_USER_ID = os.getenv("LINE_USER_ID", "").strip()
APP_URL = "https://streamlit.app"

def send_line(msg):
    if not LINE_TOKEN or not LINE_USER_ID:
        print("⚠️ 未設定 LINE 金鑰，略過推播")
        return
    api_endpoint = "https://line.me"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": msg}]
    }
    res = requests.post(api_endpoint, headers=headers, json=payload, timeout=15)
    if res.status_code == 200:
        print("✅ LINE 推播成功！")
    else:
        print(f"❌ 推播失敗 ({res.status_code}) | URL: {res.url} | 回應: {res.text}")

def get_active_stocks(min_lots=1500, min_price=15.0):
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

def scan_market(roll_win=35, rr_min=1.2):
    candidates = get_active_stocks()
    if not candidates:
        return [], []

    tickers = [c["ticker"] for c in candidates]
    name_map = {c["ticker"]: c["title"] for c in candidates}
    data = yf.download(tickers, period="6mo", group_by="ticker", progress=False, threads=True)
    
    trend_list = []   # 順勢主升段模式
    dip_list = []     # 高點後修正低接模式

    for t in tickers:
        try:
            df = data[t].dropna()
            if len(df) < 65:
                continue

            sub = df.tail(roll_win)
            curr = float(sub["Close"].iat[-1])
            hi, lo = float(sub["High"].max()), float(sub["Low"].min())
            diff = hi - lo
            if diff <= 0:
                continue

            peak_date = sub["High"].idxmax().strftime("%m/%d")
            f382, f500, f618 = hi - diff * 0.382, hi - diff * 0.500, hi - diff * 0.618
            ext1272 = hi + diff * 0.272

            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = float((100 - 100 / (1 + gain / loss)).iat[-1])

            tr = np.max(pd.concat([
                df["High"] - df["Low"],
                np.abs(df["High"] - df["Close"].shift()),
                np.abs(df["Low"] - df["Close"].shift())
            ], axis=1), axis=1)
            atr = float(tr.rolling(14).mean().iat[-1])

            ma20 = float(df["Close"].rolling(20).mean().iat[-1])
            ma60 = float(df["Close"].rolling(60).mean().iat[-1])

            if curr >= ma60 and ma20 >= ma60:
                # 模式一：順勢主升段（回測 0.382 / 月線淺支撐）
                anchor = max(f382, ma20)
                b_hi = anchor + min(0.15 * atr, anchor * 0.005)
                b_lo = anchor - min(0.25 * atr, anchor * 0.015)
                if not (b_lo * 0.985 <= curr <= b_hi * 1.025 and 40 <= rsi <= 68):
                    continue
                sl = min(f500, ma60) - 0.3 * atr
                tp1, tp2 = hi, ext1272
                tp_tag = "波段前高 / 1.272延伸"
                avg_b = (b_hi + b_lo) / 2
                rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 0
                if rr >= rr_min:
                    trend_list.append({
                        "title": name_map[t], "price": curr, "peak_date": peak_date,
                        "b_lo": b_lo, "b_hi": b_hi, "sl": sl,
                        "tp1": tp1, "tp2": tp2, "tp_tag": tp_tag, "rsi": rsi, "rr": rr
                    })
            else:
                # 模式二：高點轉折修正後回測 0.618 支撐
                anchor = f618
                b_hi = anchor + min(0.15 * atr, anchor * 0.005)
                b_lo = max(lo, anchor - min(0.25 * atr, anchor * 0.015))
                if not (b_lo * 0.98 <= curr <= b_hi * 1.03 and rsi < 48):
                    continue
                sl = lo - 0.5 * atr
                tp1, tp2 = min(ma20, f500), f382
                if tp1 > tp2:
                    tp1, tp2 = tp2, tp1
                tp_tag = "0.500反彈 / 0.382減碼"
                avg_b = (b_hi + b_lo) / 2
                rr = (tp2 - avg_b) / (avg_b - sl) if avg_b > sl else 0
                if rr >= rr_min:
                    dip_list.append({
                        "title": name_map[t], "price": curr, "peak_date": peak_date,
                        "b_lo": b_lo, "b_hi": b_hi, "sl": sl,
                        "tp1": tp1, "tp2": tp2, "tp_tag": tp_tag, "rsi": rsi, "rr": rr
                    })
        except Exception:
            continue

    trend_list.sort(key=lambda x: x["rr"], reverse=True)
    dip_list.sort(key=lambda x: x["rr"], reverse=True)
    return trend_list[:4], dip_list[:4]

def format_items(items, start_idx):
    blocks = []
    for idx, s in enumerate(items, start_idx):
        blocks.append(
            f"{idx}. {s['title']}（現價 {s['price']:.1f}｜高點 {s['peak_date']}）\n"
            f"   🟢 動態買進：{s['b_lo']:.1f} ～ {s['b_hi']:.1f} 元\n"
            f"      （第一批：{s['b_hi']:.1f} ｜ 第二批：{s['b_lo']:.1f}）\n"
            f"   🔴 移動停損：{s['sl']:.1f} 元\n"
            f"   🎯 修正賣點（{s['tp_tag']}）：\n"
            f"      第一賣點 {s['tp1']:.1f} 元 ｜ 第二賣點 {s['tp2']:.1f} 元\n"
            f"   📊 波段賺賠比：{s['rr']:.1f} 倍 ｜ RSI：{s['rsi']:.1f}"
        )
    return blocks

if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    trends, dips = scan_market()
    
    if not trends and not dips:
        msg = (
            f"📊 【AI 動態波浪選股日報】{today}\n\n"
            f"今日全市場掃描完畢，目前無符合「動態滾動支撐 + 賺賠比≥1.2」之標的，建議保留現金觀望。\n\n"
            f"📱 手機看盤 APP：{APP_URL}"
        )
    else:
        lines = [
            f"🚀 【AI 動態波浪黃金分割選股日報】\n"
            f"日期：{today} ｜ 35日滾動防線 | 賺賠比≥1.2\n" + "=" * 22
        ]
        if trends:
            lines.append("🔥 【順勢主升段】（站穩季線，回測0.382/月線）")
            lines.extend(format_items(trends, 1))
        if dips:
            lines.append("🛡️ 【高點轉折修正】（回測0.618，反彈看動態賣點）")
            lines.extend(format_items(dips, len(trends) + 1))
        lines.append(f"📱 點擊開啟動態防線圖表 APP：\n{APP_URL}")
        msg = "\n\n".join(lines)

    print(msg)
    send_line(msg)
