import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 對齊你原本設定的環境變數名稱
LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://finmindtrade.com"

def fetch_finmind(dataset, start_date, data_id=None):
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "token": FINMIND_TOKEN
    }
    if data_id:
        params["data_id"] = data_id
    res = requests.get(FINMIND_URL, params=params).json()
    if res.get("msg") != "success":
        print(f"FinMind Error ({dataset}):", res.get("msg"))
        return pd.DataFrame()
    return pd.DataFrame(res["data"])

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_screener():
    start_date = (datetime.now() - timedelta(days=110)).strftime("%Y-%m-%d")
    inst_start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    print("1. 取得台股基本資料...")
    info_df = fetch_finmind("TaiwanStockInfo", start_date)
    if info_df.empty:
        return []
    info_df = info_df[info_df["type"].isin(["twse", "tpex"])]
    info_df = info_df[info_df["stock_id"].str.len() == 4]
    stock_names = dict(zip(info_df["stock_id"], info_df["stock_name"]))
    valid_ids = set(stock_names.keys())

    print("2. 取得全市場歷史價量...")
    price_df = fetch_finmind("TaiwanStockPrice", start_date)
    if price_df.empty:
        return []
    price_df = price_df[price_df["stock_id"].isin(valid_ids)].copy()
    price_df = price_df.sort_values(["stock_id", "date"])

    print("3. 取得近5日三大法人買賣超...")
    inst_df = fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", inst_start_date)
    inst_net_5d = {}
    if not inst_df.empty:
        inst_df = inst_df[inst_df["stock_id"].isin(valid_ids)].copy()
        inst_df["net"] = inst_df["buy"] - inst_df["sell"]
        recent_dates = sorted(inst_df["date"].unique())[-5:]
        inst_recent = inst_df[inst_df["date"].isin(recent_dates)]
        inst_net_5d = (inst_recent.groupby("stock_id")["net"].sum() / 1000).to_dict()

    print("4. 計算全市場技術指標（下跌找買點、上漲找賣點）...")
    candidates = []

    for stock_id, group in price_df.groupby("stock_id"):
        if len(group) < 62:
            continue

        close = group["close"]
        open_p = group["open"]
        volume = group["Trading_Volume"] / 1000

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol_ma5 = volume.rolling(5).mean()
        rsi14 = calculate_rsi(close, 14)

        curr_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        curr_open = open_p.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        curr_ma60 = ma60.iloc[-1]
        curr_vol_ma5 = vol_ma5.iloc[-1]
        curr_rsi = rsi14.iloc[-1]
        prev_rsi = rsi14.iloc[-2]
        high_20d = group["max"].iloc[-20:].max()

        # 1. 流動性濾網：5日均量 > 1000 張，股價 > 15 元
        if curr_vol_ma5 < 1000 or curr_close < 15:
            continue
        # 2. 長線保護短線：收盤價在 60日季線之上（多頭回檔，非空頭破底）
        if curr_close <= curr_ma60:
            continue
        # 3. 股價跌破月線 (MA20)，代表短線確實有拉回找買點
        if curr_close >= curr_ma20:
            continue
        # 4. 超賣止跌訊號：14日 RSI < 40，且今日 RSI 勾頭或收紅K
        is_rebounding = (curr_rsi >= prev_rsi) or (curr_close > curr_open)
        if not (curr_rsi < 40 and is_rebounding):
            continue
        # 5. 籌碼濾網：近 5 日三大法人合計買超 > 0 張
        net_5d = inst_net_5d.get(stock_id, 0)
        if net_5d <= 0:
            continue

        upside_to_ma20 = ((curr_ma20 - curr_close) / curr_close) * 100
        upside_to_high = ((high_20d - curr_close) / curr_close) * 100
        pct_change = ((curr_close - prev_close) / prev_close) * 100

        candidates.append({
            "stock_id": stock_id,
            "name": stock_names.get(stock_id, ""),
            "close": curr_close,
            "pct": pct_change,
            "rsi": curr_rsi,
            "ma20": curr_ma20,
            "ma60": curr_ma60,
            "high_20d": high_20d,
            "upside_ma20": upside_to_ma20,
            "upside_high": upside_to_high,
            "net_5d": net_5d
        })

    df_res = pd.DataFrame(candidates)
    if df_res.empty:
        return []

    # 依「RSI 最低（最超賣）」與「法人買超最多」排序，精選前 5 檔
    df_res = df_res.sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(5)
    return df_res.to_dict("records")

def send_line_message(text):
    url = "https://line.me"
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
