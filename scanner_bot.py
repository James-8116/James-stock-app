import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. LINE Messaging API 與專屬 APP 網址設定
# ==========================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_TOKEN", "cZtRmdtBZWhb/e9TYjOi7EpviEid8MzxWkLgzBVe8xxdn5mR75wHMG9rEchG7X8ydWCMTHKiUtw8VU8//sBPZR6dWbADRK7eAtgDzeG93/Sd44xqdm21erBISKZMasyu8KxzmYnb0jxLB9yx8g3ddwdB04t89/1O/w1cDnyilFU=")
LINE_USER_ID = os.getenv("LINE_USER_ID", "U81b3432bfd33263e5b273a8463e66393")
MY_APP_URL = "https://streamlit.app"

def send_line_push(message_text):
    """透過 LINE Messaging API 發送主動推播訊息"""
    url = "https://line.me"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message_text}]
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        print("✅ LINE 訊息推播成功！")
    else:
        print(f"❌ LINE 推播失敗：{res.status_code} - {res.text}")

# ==========================================
# 2. 第一階段：抓取全市場活躍股清單（TWSE + TPEx）
# ==========================================
def get_active_stocks(min_volume_lots=1000, min_price=15.0):
    """取得當日成交量大於 1,000 張且股價大於 15 元的上市櫃股票"""
    active_list = []
    
    # 抓取上市股票 (TWSE OpenAPI)
    try:
        twse_url = "https://twse.com.tw"
        twse_data = requests.get(twse_url, timeout=15).json()
        for item in twse_data:
            code = item.get("Code", "").strip()
            name = item.get("Name", "").strip()
            if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                try:
                    vol_lots = int(item.get("TradeVolume", "0")) // 1000
                    close_p = float(item.get("ClosingPrice", "0"))
                    if vol_lots >= min_volume_lots and close_p >= min_price:
                        active_list.append({"ticker": f"{code}.TW", "code": code, "name": name})
                except ValueError:
                    continue
    except Exception as e:
        print(f"上市資料抓取異常：{e}")

    # 抓取上櫃股票 (TPEx OpenAPI)
    try:
        tpex_url = "https://tpex.org.tw"
        tpex_data = requests.get(tpex_url, timeout=15).json()
        for item in tpex_data:
            code = item.get("SecuritiesCompanyCode", "").strip()
            name = item.get("CompanyName", "").strip()
            if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                try:
                    vol_lots = int(str(item.get("TradingShares", "0")).replace(",", "")) // 1000
                    close_p = float(str(item.get("Close", "0")).replace(",", ""))
                    if vol_lots >= min_volume_lots and close_p >= min_price:
                        active_list.append({"ticker": f"{code}.TWO", "code": code, "name": name})
                except ValueError:
                    continue
    except Exception as e:
        print(f"上櫃資料抓取異常：{e}")

    return active_list

# ==========================================
# 3. 第二階段：波浪黃金分割 + 建議買賣價區間精算
# ==========================================
def scan_market(lookback_days=90, rr_min=2.0):
    candidates = get_active_stocks(min_volume_lots=1000, min_price=15.0)
    print(f"🔍 初篩完成，共 {len(candidates)} 檔活躍股進入技術面精算...")
    
    if not candidates:
        return []

    tickers = [c["ticker"] for c in candidates]
    name_map = {c["ticker"]: f"{c['code']} {c['name']}" for c in candidates}
    
    data = yf.download(tickers, period="1y", group_by="ticker", progress=False, threads=True)
    matched_stocks = []
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < lookback_days:
                continue
                
            sub_df = df.tail(lookback_days)
            curr_p = float(sub_df["Close"].iloc[-1])
            s_high = float(sub_df["High"].max())
            s_low = float(sub_df["Low"].min())
            diff = s_high - s_low
            if diff <= 0:
                continue
                
            # 黃金分割關鍵點位（針對波段高低點計算）
            fib_382 = s_high - diff * 0.382  # 弱勢反彈位（第一停利）
            fib_500 = s_high - diff * 0.500  # 中軸平衡位
            fib_618 = s_high - diff * 0.618  # 黃金買進支撐位
            
            # 條件 A：股價正處於 0.618 黃金買進支撐區附近（上下 3% 緩衝區）
            if not (fib_618 * 0.97 <= curr_p <= fib_618 * 1.03):
                continue
                
            # 計算 14 日 RSI 動能
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # 條件 B：RSI 處於相對低檔超賣或止跌轉折區（< 42）
            if rsi >= 42:
                continue
                
            # 計算 14 日 ATR（真實波動幅度）設定動態買賣區間與停損
            tr = pd.concat([
                df["High"] - df["Low"],
                np.abs(df["High"] - df["Close"].shift()),
                np.abs(df["Low"] - df["Close"].shift())
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            
            # 建議分批買進區間：以 0.618 支撐位為上緣，向下延伸 0.5 倍 ATR（但不低於波段低點）
            buy_high = max(fib_618, curr_p)
            buy_low = max(s_low, fib_618 - 0.5 * atr)
            
            # 防洗盤嚴格停損價：波段低點下方 0.5 倍 ATR
            stop_loss = s_low - 0.5 * atr
            
            # 雙重停利目標計算（針對從低點反彈至高點波段）
            tp1 = fib_382  # 第一停利位（0.382）
            tp2 = s_high - diff * 0.236 if fib_500 <= curr_p else fib_500  # 第二停利位（0.500 或 0.236）
            
            # 計算賺賠比（以建議買進區間中值進場、第一停利位出場計算）
            avg_buy = (buy_high + buy_low) / 2
            risk = avg_buy - stop_loss
            reward = tp1 - avg_buy
            rr_ratio = reward / risk if risk > 0 else 0
            
            # 條件 C：風險報酬比必須大於設定門檻（預設 2.0 倍）
            if rr_ratio >= rr_min:
                matched_stocks.append({
                    "title": name_map[ticker],
                    "price": curr_p,
                    "buy_low": buy_low,
                    "buy_high": buy_high,
                    "stop_loss": stop_loss,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rsi": rsi,
                    "rr": rr_ratio
                })
        except Exception:
            continue
            
    matched_stocks.sort(key=lambda x: x["rr"], reverse=True)
    return matched_stocks[:8]

# ==========================================
# 4. 主程式執行與買賣建議報告組裝
# ==========================================
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    results = scan_market(lookback_days=90, rr_min=2.0)
    
    if not results:
        report = (
            f"📊 【AI 量化選股日報】{today_str}\n\n"
            f"今日全市場掃描完畢，目前無同時符合「回測 0.618 支撐 + RSI低檔 + 賺賠比 ≥ 2」之標的，建議保留現金觀望。\n\n"
            f"📱 手機看盤 APP：{MY_APP_URL}"
        )
    else:
        lines = [
            f"🚀 【AI 波浪黃金分割買賣建議日報】\n"
            f"日期：{today_str}\n"
            f"策略：回測0.618支撐 | RSI<42 | 賺賠比≥2\n" + "=" * 22
        ]
        for idx, s in enumerate(results, 1):
            lines.append(
                f"{idx}. {s['title']}（現價 {s['price']:.1f} 元）\n"
                f"   🟢 建議買進區間：{s['buy_low']:.1f} ～ {s['buy_high']:.1f} 元\n"
                f"   🔴 防洗盤停損價：{s['stop_loss']:.1f} 元\n"
                f"   🎯 第一停利 (0.382)：{s['tp1']:.1f} 元\n"
                f"   🏆 第二停利 (0.500)：{s['tp2']:.1f} 元\n"
                f"   📊 賺賠比：{s['rr']:.1f} 倍 ｜ RSI：{s['rsi']:.1f}"
            )
        lines.append(f"📱 點擊開啟互動圖表 APP：\n{MY_APP_URL}")
        report = "\n\n".join(lines)
        
    print(report)
    send_line_push(report)
