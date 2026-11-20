import os
import time
import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 關閉證交所憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://finmindtrade.com"

def get_market_snapshot():
    """從證交所與櫃買中心官方 API 取得全市場最新收盤行情（免 Token、極速）"""
    candidates = {}

    # 1. 上市股票 (TWSE)
    try:
        r_twse = requests.get(
            "https://twse.com.tw",
            verify=False, timeout=15
        )
        if r_twse.status_code == 200:
            for item in r_twse.json():
                code = item.get("Code", "").strip()
                if len(code) == 4 and code.isdigit():
                    try:
                        vol_lots = int(item.get("TradeVolume", "0").replace(",", "")) / 1000
                        close_p = float(item.get("ClosingPrice", "0").replace(",", ""))
                        # 初步濾網：單日成交量 > 800 張、股價 > 15 元
                        if vol_lots >= 800 and close_p >= 15:
                            candidates[code] = item.get("Name", "").strip()
                    except ValueError:
                        continue
    except Exception as e:
        print("TWSE API Error:", e)

    # 2. 上櫃股票 (TPEx)
    try:
        r_tpex = requests.get(
            "https://tpex.org.tw",
            verify=False, timeout=15
        )
        if r_tpex.status_code == 200:
            for item in r_tpex.json():
                code = item.get("SecuritiesCompanyCode", "").strip()
                if len(code) == 4 and code.isdigit():
                    try:
                        vol_lots = int(item.get("TradingShares", "0").replace(",", "")) / 1000
                        close_p = float(item.get("Close", "0").replace(",", ""))
                        if vol_lots >= 800 and close_p >= 15:
                            candidates[code] = item.get("CompanyName", "").strip()
                    except ValueError:
                        continue
    except Exception as e:
        print("TPEx API Error:", e)

    return candidates

def fetch_finmind_stock(dataset, stock_id, start_date):
    """帶入指定股票代號向 FinMind 查詢歷史資料（合規且穩定）"""
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "token": FINMIND_TOKEN
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10)
        if res.status_code != 200 or not res.text.strip():
            return pd.DataFrame()
        data = res.json()
        if data.get("msg") != "success":
            return pd.DataFrame()
        return pd.DataFrame(data["data"])
    except Exception:
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_screener():
    print("1. 取得上市櫃全市場行情快照...")
    pool = get_market_snapshot()
    print(f"初步符合流動性條件之標的共 {len(pool)} 檔")

    if not pool:
        return []

    start_date = (datetime.now() - timedelta(days=110)).strftime("%Y-%m-%d")
    inst_start_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")

    technical_passed = []

    print("2. 逐檔計算技術指標（季線保護 + 月線下回檔 + RSI超賣止跌）...")
    for idx, (stock_id, name) in enumerate(pool.items()):
        df = fetch_finmind_stock("TaiwanStockPrice", stock_id, start_date)
        if df.empty or len(df) < 62:
            continue

        close = df["close"]
        open_p = df["open"]
        volume = df["Trading_Volume"] / 1000

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
        high_20d = float(df["max"].iloc[-20:].max())

        # 5日均量 > 1000 張
        if curr_vol_ma5 < 1000:
            continue
        # 長線保護短線：收盤價在 60日季線之上
        if curr_close <= curr_ma60:
            continue
        # 短線回檔：股價在 20日月線之下
        if curr_close >= curr_ma20:
            continue
        # 下跌找買點：RSI < 40 且止跌勾頭（今日RSI >= 昨日RSI 或 今日收紅K）
        is_rebounding = (curr_rsi >= prev_rsi) or (curr_close > curr_open)
        if not (curr_rsi < 40 and is_rebounding):
            continue

        technical_passed.append({
            "stock_id": stock_id,
            "name": name,
            "close": curr_close,
            "pct": ((curr_close - prev_close) / prev_close) * 100,
            "rsi": curr_rsi,
            "ma20": curr_ma20,
            "ma60": curr_ma60,
            "high_20d": high_20d,
            "upside_ma20": ((curr_ma20 - curr_close) / curr_close) * 100,
            "upside_high": ((high_20d - curr_close) / curr_close) * 100,
        })

        # 避免瞬間請求過快，每查詢 20 檔稍停 0.5 秒
        if (idx + 1) % 20 == 0:
            time.sleep(0.5)

    print(f"3. 技術面過關共 {len(technical_passed)} 檔，開始檢查近5日三大法人買賣超...")
    final_candidates = []
    for item in technical_passed:
        inst_df = fetch_finmind_stock("TaiwanStockInstitutionalInvestorsBuySell", item["stock_id"], inst_start_date)
        if inst_df.empty:
            continue
        inst_df["net"] = inst_df["buy"] - inst_df["sell"]
        recent_dates = sorted(inst_df["date"].unique())[-5:]
        net_5d = inst_df[inst_df["date"].isin(recent_dates)]["net"].sum() / 1000

        # 籌碼濾網：近 5 日三大法人合計買超 > 0 張
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
