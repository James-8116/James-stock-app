import os
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"

def get_tw_stock_list():
    params = {"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        df = df[df["type"].isin(["twse", "tpex"]) & (df["stock_id"].str.len() == 4) & (df["stock_id"].str.isdigit())]
        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {"id": r["stock_id"], "name": r["stock_name"]}
            for _, r in df.iterrows()
        }
    except Exception as e:
        print("取得股票清單失敗:", e)
        return {}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def check_fib_touch(curr_close, curr_low, high_60d, low_60d, tol=0.015):
    """計算近60日波段黃金分割率 (0.382 / 0.5 / 0.618) 並判斷股價是否碰觸支撐"""
    swing = high_60d - low_60d
    if swing <= 0:
        return None, 0.0
    fibs = {
        "0.382": high_60d - swing * 0.382,
        "0.500": high_60d - swing * 0.500,
        "0.618": high_60d - swing * 0.618,
    }
    for k, v in fibs.items():
        if abs(curr_close - v) / v <= tol or abs(curr_low - v) / v <= tol:
            return k, v
    return None, 0.0

def check_inst_buy_5d(stock_id):
    start = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start, "token": FINMIND_TOKEN}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10).json()
        if res.get("msg") != "success" or not res.get("data"):
            return 0
        df = pd.DataFrame(res["data"])
        df["net"] = df["buy"] - df["sell"]
        recent = sorted(df["date"].unique())[-5:]
        return df[df["date"].isin(recent)]["net"].sum() / 1000
    except Exception:
        return 0

def run_screener():
    stock_map = get_tw_stock_list()
    tickers = list(stock_map.keys())
    if not tickers:
        return []

    passed = []
    batch_size = 250
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].dropna(subset=["Close", "Volume"])
                if len(df) < 62:
                    continue

                close, low_p = df["Close"], df["Low"]
                vol_ma5 = (df["Volume"] / 1000).rolling(5).mean().iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma60 = close.rolling(60).mean().iloc[-1]
                rsi = calculate_rsi(close, 14)

                curr_c, prev_c, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(low_p.iloc[-1])
                curr_rsi = float(rsi.iloc[-1])
                high_20d = float(df["High"].iloc[-20:].max())
                high_60d, low_60d = float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())

                # 1. 流動性與多頭保護（季線之上）
                if vol_ma5 < 1000 or curr_c < 15 or curr_c <= ma60:
                    continue
                # 2. 高點拉回至少 3%
                pullback = ((high_20d - curr_c) / high_20d) * 100
                if pullback < 3.0:
                    continue
                # 3. 碰觸黃金分割率 (0.382 / 0.500 / 0.618，容許誤差 ±1.5%)
                fib_hit, fib_p = check_fib_touch(curr_c, curr_l, high_60d, low_60d, 0.015)
                if not fib_hit:
                    continue
                # 4. RSI 低檔濾網 (< 48)
                if curr_rsi >= 48:
                    continue

                info = stock_map[t]
                passed.append({
                    "stock_id": info["id"], "name": info["name"],
                    "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
                    "rsi": curr_rsi, "fib_hit": fib_hit, "fib_price": fib_p,
                    "ma20": ma20, "ma60": ma60, "high_20d": high_20d,
                    "up_ma20": ((ma20 - curr_c) / curr_c) * 100,
                    "up_high": ((high_20d - curr_c) / curr_c) * 100
                })
        except Exception as e:
            print(f"批次 {i} 異常:", e)
        time.sleep(0.5)

    # 依 RSI 最低排序取前 40 檔檢查三大法人買超
    passed = sorted(passed, key=lambda x: x["rsi"])[:40]
    final_list = []
    for item in passed:
        net = check_inst_buy_5d(item["stock_id"])
        if net > 0:
            item["net_5d"] = int(net)
            final_list.append(item)

    if not final_list:
        return []

    df_res = pd.DataFrame(final_list).sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(5)
    return df_res.to_dict("records")

def send_line_message(text):
    if not LINE_TOKEN or not LINE_USER_ID:
        return
    url = "https://api." + "line" + ".me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": text}]}
    res = requests.post(url, headers=headers, json=payload)
    print("LINE Push Status:", res.status_code, res.text)

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    top5 = run_screener()

    if not top5:
        msg = (
            f"📉 【台股黃金分割波段選股】{today_str}\n"
            f"策略：季線上回檔 + 黃金分割支撐 + 法人低接\n\n"
            f"今日全市場無同時符合「黃金分割支撐」與「法人買超」之標的。"
        )
    else:
        lines = [
            f"📉 【台股黃金分割精選 TOP 5】{today_str}",
            "策略：季線上回檔 + 碰黃金分割位 + 5日法人買超\n"
        ]
        for idx, s in enumerate(top5, 1):
            block = (
                f"{idx}. {s['name']} ({s['stock_id']})  現價: {s['close']:.1f} ({s['pct']:+.2f}%)\n"
                f"   📐 黃金支撐：{s['fib_hit']} ({s['fib_price']:.1f}) | RSI {s['rsi']:.1f}\n"
                f"   🟢 籌碼低接：5日三大法人 +{s['net_5d']:,} 張\n"
                f"   🔴 賣點參考：月線 {s['ma20']:.1f} ({s['up_ma20']:+.1f}%) / 前高 {s['high_20d']:.1f} (+{s['up_high']:.1f}%)\n"
                f"   🛡️ 防守點位：季線 {s['ma60']:.1f} (跌破停損)"
            )
            lines.append(block)
        lines.append("\n⚠️ 量化篩選僅供研究，進場前請確認無重大基本面利空。")
        msg = "\n".join(lines)

    send_line_message(msg)
