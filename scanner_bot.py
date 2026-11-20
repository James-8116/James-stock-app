import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"

def get_tw_stock_list():
    """從 FinMind 取得台股上市櫃股票代號與名稱（不帶日期參數，穩定不超載）"""
    params = {
        "dataset": "TaiwanStockInfo",
        "token": FINMIND_TOKEN
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=15)
        data = res.json()
        if data.get("msg") != "success":
            print("FinMind StockInfo Error:", data.get("msg"))
            return {}
        df = pd.DataFrame(data["data"])
        # 只保留上市 (twse) 與上櫃 (tpex) 的 4 碼一般股票
        df = df[df["type"].isin(["twse", "tpex"])]
        df = df[df["stock_id"].str.len() == 4]
        df = df[df["stock_id"].str.isdigit()]
        # 加上 yfinance 後綴 (.TW 或 .TWO)
        stock_map = {}
        for _, row in df.iterrows():
            suffix = ".TW" if row["type"] == "twse" else ".TWO"
            stock_map[row["stock_id"] + suffix] = {
                "id": row["stock_id"],
                "name": row["stock_name"]
            }
        return stock_map
    except Exception as e:
        print("取得股票清單失敗:", e)
        return {}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_inst_buy_5d(stock_id):
    """查詢單一股票近 5 日三大法人合計買賣超 (張)"""
    start_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date,
        "token": FINMIND_TOKEN
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10)
        data = res.json()
        if data.get("msg") != "success" or not data.get("data"):
            return 0
        df = pd.DataFrame(data["data"])
        df["net"] = df["buy"] - df["sell"]
        recent_dates = sorted(df["date"].unique())[-5:]
        return df[df["date"].isin(recent_dates)]["net"].sum() / 1000
    except Exception:
        return 0

def run_screener():
    print("1. 取得上市櫃股票清單...")
    stock_map = get_tw_stock_list()
    tickers = list(stock_map.keys())
    print(f"共取得 {len(tickers)} 檔上市櫃一般股票")

    if not tickers:
        return []

    print("2. 透過 Yahoo Finance 批次下載近 4 個月日K線（美國主機高速連線）...")
    technical_passed = []
    batch_size = 250

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
            for ticker in batch:
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].dropna(subset=["Close", "Volume"])
                if len(df) < 62:
                    continue

                close = df["Close"]
                open_p = df["Open"]
                volume = df["Volume"] / 1000  # 換算為「張」

                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean()
                vol_ma5 = volume.rolling(5).mean()
                rsi14 = calculate_rsi(close, 14)

                curr_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                curr_open = float(open_p.iloc[-1])
                curr_ma20 = float(ma20.iloc[-1])
                curr_ma60 = float(ma60.iloc[-1])
                curr_vol_ma5 = float(vol_ma5.iloc[-1])
                curr_rsi = float(rsi14.iloc[-1])
                prev_rsi = float(rsi14.iloc[-2])
                high_20d = float(df["High"].iloc[-20:].max())

                # 1. 流動性濾網：5日均量 > 1000 張，股價 > 15 元
                if curr_vol_ma5 < 1000 or curr_close < 15:
                    continue
                # 2. 長線保護短線：收盤價在 60日季線之上
                if curr_close <= curr_ma60:
                    continue
                # 3. 短線回檔：股價在 20日月線之下
                if curr_close >= curr_ma20:
                    continue
                # 4. 下跌找買點：RSI < 40 且止跌勾頭（今日RSI >= 昨日RSI 或 今日收紅K）
                is_rebounding = (curr_rsi >= prev_rsi) or (curr_close > curr_open)
                if not (curr_rsi < 40 and is_rebounding):
                    continue

                info = stock_map[ticker]
                technical_passed.append({
                    "stock_id": info["id"],
                    "name": info["name"],
                    "close": curr_close,
                    "pct": ((curr_close - prev_close) / prev_close) * 100,
                    "rsi": curr_rsi,
                    "ma20": curr_ma20,
                    "ma60": curr_ma60,
                    "high_20d": high_20d,
                    "upside_ma20": ((curr_ma20 - curr_close) / curr_close) * 100,
                    "upside_high": ((high_20d - curr_close) / curr_close) * 100,
                })
        except Exception as e:
            print(f"批次 {i} 下載異常:", e)
        time.sleep(1)

    print(f"3. 技術面過關共 {len(technical_passed)} 檔，檢查近 5 日三大法人買超...")
    final_candidates = []
    for item in technical_passed:
        net_5d = check_inst_buy_5d(item["stock_id"])
        if net_5d > 0:
            item["net_5d"] = net_5d
            final_candidates.append(item)

    if not final_candidates:
        return []

    df_res = pd.DataFrame(final_candidates)
    df_res = df_res.sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(5)
    return df_res.to_dict("records")

def send_line_message(text):
    if not LINE_TOKEN or not LINE_USER_ID:
        print("缺少 LINE_TOKEN 或 LINE_USER_ID 環境變數")
        return
    # 修正 405 錯誤：使用最新 LINE Messaging API 推播網址
    url = "https://api." + "line" + ".me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": text}]
    }
    res = requests.post(url, headers=headers, json=payload)
    print("LINE Push Status:", res.status_code, res.text)

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    top5 = run_screener()

    if not top5:
        msg = (
            f"📉 【台股逆勢波段選股】{today_str}\n"
            f"策略：季線上回檔 + RSI超賣止跌 + 法人低接\n\n"
            f"今日全市場無同時符合「技術超賣」與「法人買超」之標的。"
        )
    else:
        lines = [
            f"📉 【台股逆勢波段精選 TOP 5】{today_str}",
            "策略：季線上回檔 + RSI<40止跌 + 5日法人買超\n"
        ]
        for idx, s in enumerate(top5, 1):
            block = (
                f"{idx}. {s['name']} ({s['stock_id']})  現價: {s['close']:.1f} ({s['pct']:+.2f}%)\n"
                f"   🟢 買點訊號：RSI {s['rsi']:.1f} 勾頭 | 5日法人 +{int(s['net_5d']):,} 張\n"
                f"   🔴 賣點參考：月線 {s['ma20']:.1f} (+{s['upside_ma20']:.1f}%) / 前高 {s['high_20d']:.1f} (+{s['upside_high']:.1f}%)\n"
                f"   🛡️ 防守點位：季線 {s['ma60']:.1f} (跌破停損)"
            )
            lines.append(block)
        lines.append("\n⚠️ 量化篩選僅供研究，進場前請確認無重大基本面利空。")
        msg = "\n".join(lines)

    send_line_message(msg)
