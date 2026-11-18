import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

LINE_TOKEN = os.getenv("LINE_TOKEN", "").strip()
LINE_USER_ID = os.getenv("LINE_USER_ID", "").strip()
APP_URL = "https://" + "james-stockapp" + ".streamlit.app/"

# 內建台股 120 檔高流動性主流上市櫃股票（100% 穩定顯示中文股名，不依賴外部總表）
WATCH_POOL = {
    "2330.TW": "2330 台積電", "2317.TW": "2317 鴻海", "2454.TW": "2454 聯發科", "2308.TW": "2308 台達電",
    "2382.TW": "2382 廣達", "3231.TW": "3231 緯創", "6669.TW": "6669 緯穎", "2303.TW": "2303 聯電",
    "3711.TW": "3711 日月光", "2357.TW": "2357 華碩", "3008.TW": "3008 大立光", "3019.TW": "3019 亞光",
    "2340.TW": "2340 台亞", "3450.TW": "3450 聯鈞", "6442.TW": "6442 光聖", "2379.TW": "2379 瑞昱",
    "3034.TW": "3034 聯詠", "2327.TW": "2327 國巨", "2376.TW": "2376 技嘉", "2301.TW": "2301 光寶科",
    "2603.TW": "2603 長榮", "2609.TW": "2609 陽明", "2615.TW": "2615 萬海", "2618.TW": "2618 長榮航",
    "2610.TW": "2610 華航", "2881.TW": "2881 富邦金", "2882.TW": "2882 國泰金", "2891.TW": "2891 中信金",
    "2886.TW": "2886 兆豐金", "2884.TW": "2884 玉山金", "2885.TW": "2885 元大金", "2890.TW": "2890 永豐金",
    "2002.TW": "2002 中鋼", "1101.TW": "1101 台泥", "1301.TW": "1301 台塑", "1303.TW": "1303 南亞",
    "1513.TW": "1513 中興電", "1519.TW": "1519 華城", "1503.TW": "1503 士電", "1504.TW": "1504 東元",
    "2356.TW": "2356 英業達", "2353.TW": "2353 宏碁", "2324.TW": "2324 仁寶", "4938.TW": "4938 和碩",
    "2383.TW": "2383 台光電", "3037.TW": "3037 欣興", "3189.TW": "3189 景碩", "8046.TW": "8046 南電",
    "2368.TW": "2368 金像電", "6213.TW": "6213 聯茂", "3653.TW": "3653 健策", "3017.TW": "3017 奇鋐",
    "2421.TW": "2421 建準", "3376.TW": "3376 新日興", "6805.TW": "6805 富世達", "3533.TW": "3533 嘉澤",
    "2345.TW": "2345 智邦", "2313.TW": "2313 華通", "2367.TW": "2367 燿華", "2449.TW": "2449 京元電",
    "3443.TW": "3443 創意", "3661.TW": "3661 世芯KY", "3035.TW": "3035 智原", "2388.TW": "2388 威盛",
    "2408.TW": "2408 南亞科", "2344.TW": "2344 華邦電", "2337.TW": "2337 旺宏", "3702.TW": "3702 大聯大",
    "3036.TW": "3036 文曄", "2395.TW": "2395 研華", "2049.TW": "2049 上銀", "1590.TW": "1590 亞德客",
    "2207.TW": "2207 和泰車", "2912.TW": "2912 統一超", "1216.TW": "1216 統一", "9910.TW": "9910 豐泰",
    "3324.TWO": "3324 雙鴻", "4908.TWO": "4908 前鼎", "3081.TWO": "3081 聯亞", "3363.TWO": "3363 上詮",
    "3163.TWO": "3163 波若威", "4979.TWO": "4979 華星光", "8069.TWO": "8069 元太", "8299.TWO": "8299 群聯",
    "6488.TWO": "6488 環球晶", "5483.TWO": "5483 中美晶", "5347.TWO": "5347 世界", "3529.TWO": "3529 力旺",
    "5274.TWO": "5274 信驊", "6147.TWO": "6147 頎邦", "3260.TWO": "3260 威剛", "8086.TWO": "8086 宏捷科",
    "3105.TWO": "3105 穩懋", "6274.TWO": "6274 台燿", "5425.TWO": "5425 台半", "6182.TWO": "6182 合晶",
    "3293.TWO": "3293 鈊象", "6472.TWO": "6472 保瑞", "1795.TWO": "1795 美時", "4147.TWO": "4147 中裕"
}

def send_line(msg):
    if not LINE_TOKEN or not LINE_USER_ID: return
    push_url = "https://" + "api.line.me" + "/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    res = requests.post(push_url, headers=headers, json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": msg}]}, timeout=15)
    print("✅ LINE 推播成功！" if res.status_code == 200 else f"❌ 推播失敗：{res.text}")

def scan_market(roll_win=35):
    trends, dips = [], []
    for t, title in WATCH_POOL.items():
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
    return trends[:4], dips[:4]

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
